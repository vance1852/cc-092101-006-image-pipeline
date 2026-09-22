"""Tests for the resumable run ledger and crash-recovery behaviour."""

import json
import os
import time

import pytest

from image_pipeline.algorithms import core as alg
from image_pipeline.batch.executor import BatchExecutor
from image_pipeline.batch.ledger import (
    Ledger,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_PROCESSING,
    config_fingerprint,
)
from image_pipeline.config.loader import PipelineConfig
from image_pipeline.utils.image_io import (
    TEMP_FILE_PREFIX,
    file_sha256,
    write_image,
)


COMPLEX_CFG = {
    'version': '1.0', 'name': 'two_outputs',
    'nodes': [
        {'id': 'in', 'type': 'input'},
        {'id': 'gray', 'type': 'grayscale'},
        {'id': 'sobel', 'type': 'sobel', 'params': {'direction': 'both'}},
        {'id': 'bright', 'type': 'brightness', 'params': {'value': 20}},
        {'id': 'edges_out', 'type': 'output', 'params': {'suffix': '_edges'}},
        {'id': 'enhanced_out', 'type': 'output',
         'params': {'format': 'JPEG', 'quality': 85, 'suffix': '_enhanced'}},
    ],
    'edges': [
        {'from': 'in', 'to': 'gray'},
        {'from': 'gray', 'to': 'sobel'},
        {'from': 'sobel', 'to': 'edges_out'},
        {'from': 'gray', 'to': 'bright'},
        {'from': 'bright', 'to': 'enhanced_out'},
    ],
}


SIMPLE_CFG = {
    'version': '1.0', 'name': 'test_pipeline',
    'nodes': [
        {'id': 'in', 'type': 'input'},
        {'id': 'gray', 'type': 'grayscale'},
        {'id': 'out', 'type': 'output', 'params': {'suffix': '_out'}},
    ],
    'edges': [{'from': 'in', 'to': 'gray'}, {'from': 'gray', 'to': 'out'}],
}


def _make_executor(config=None):
    cfg = PipelineConfig(config or SIMPLE_CFG)
    executor, validation = cfg.build_executor()
    assert validation.valid, validation.errors
    return executor


def _make_inputs(in_dir, names=('a.png', 'b.png', 'c.png'), size=8):
    os.makedirs(in_dir, exist_ok=True)
    paths = []
    for i, name in enumerate(names):
        p = os.path.join(in_dir, name)
        write_image(alg.generate_gradient_image(size + i, size + i), p, fmt='PNG')
        paths.append(p)
    return paths


def _run_batch(in_dir, out_dir, config=None, **kw):
    executor = _make_executor(config)
    raw = config or SIMPLE_CFG
    batch = BatchExecutor(executor, in_dir, out_dir, config_file='cfg.json',
                          config_raw=raw, **kw)
    return batch.run()


def _read_ledger_lines(out_dir):
    path = os.path.join(out_dir, '.imgpipe', 'ledger.jsonl')
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _states_for(lines, run_id):
    return [r['state'] for r in lines
            if r.get('type') == 'item' and r.get('run_id') == run_id]


# ---------------------------------------------------------------------------
# Basic ledger mechanics
# ---------------------------------------------------------------------------

class TestLedgerBasics:

    def test_first_run_completes_everything(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir)
        report = _run_batch(in_dir, out_dir)
        assert report.total == 3
        assert report.new_completed == 3
        assert report.reused == 0
        assert report.failed == 0
        assert os.path.isfile(os.path.join(out_dir, '.imgpipe', 'ledger.jsonl'))
        lines = _read_ledger_lines(out_dir)
        assert sum(1 for r in lines if r['type'] == 'run') == 1
        states = _states_for(lines, report.run_id)
        # Every item: processing -> completed
        assert states.count(STATE_PROCESSING) == 3
        assert states.count(STATE_COMPLETED) == 3

    def test_second_run_reuses_everything_without_recompute(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir)
        r1 = _run_batch(in_dir, out_dir)
        assert r1.new_completed == 3
        mtime_before = os.path.getmtime(os.path.join(out_dir, 'a_out.png'))
        time.sleep(0.02)
        r2 = _run_batch(in_dir, out_dir)
        assert r2.new_completed == 0
        assert r2.reused == 3
        assert r2.skipped == 3
        assert r2.failed == 0
        # Outputs untouched.
        assert os.path.getmtime(os.path.join(out_dir, 'a_out.png')) == mtime_before
        for r in r2.results:
            assert r.disposition == 'reused'
            assert r.reused is True

    def test_report_distinguishes_four_dispositions(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png', 'b.png'))
        _run_batch(in_dir, out_dir)
        # Add a corrupt file for the second run; a,b are reused, bad is invalid.
        with open(os.path.join(in_dir, 'bad.png'), 'wb') as f:
            f.write(b'not an image' * 30)
        r2 = _run_batch(in_dir, out_dir)
        disp = {os.path.basename(r.input_path): r.disposition for r in r2.results}
        assert disp['a.png'] == 'reused'
        assert disp['b.png'] == 'reused'
        assert disp['bad.png'] == 'invalid'
        assert r2.reused == 2 and r2.invalid == 1
        data = r2.to_dict()
        assert data['summary']['reused'] == 2
        assert data['summary']['invalid'] == 1


# ---------------------------------------------------------------------------
# Skip must require: input bytes + config + target files all match
# ---------------------------------------------------------------------------

class TestSkipConditions:

    def test_changed_input_bytes_forces_recompute(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        p = os.path.join(in_dir, 'a.png')
        write_image(alg.generate_gradient_image(8, 8), p, fmt='PNG')
        r1 = _run_batch(in_dir, out_dir)
        assert r1.new_completed == 1
        # Replace input bytes (same path).
        write_image(alg.generate_checkerboard(10, 10, 2), p, fmt='PNG')
        r2 = _run_batch(in_dir, out_dir)
        assert r2.new_completed == 1
        assert r2.reused == 0
        assert r2.stale == 1
        res = r2.results[0]
        assert res.disposition == 'new'
        assert res.stale is True
        assert 'input file bytes changed' in (res.error or '')

    def test_changed_config_forces_recompute(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        _run_batch(in_dir, out_dir, config=SIMPLE_CFG)
        cfg2 = json.loads(json.dumps(SIMPLE_CFG))
        cfg2['nodes'][1] = {'id': 'gray', 'type': 'grayscale',
                            'params': {}}
        # Add a threshold node to actually change pixel-affecting config.
        cfg2['nodes'] = [
            {'id': 'in', 'type': 'input'},
            {'id': 'gray', 'type': 'grayscale'},
            {'id': 'thr', 'type': 'threshold', 'params': {'value': 64}},
            {'id': 'out', 'type': 'output', 'params': {'suffix': '_out'}},
        ]
        cfg2['edges'] = [
            {'from': 'in', 'to': 'gray'},
            {'from': 'gray', 'to': 'thr'},
            {'from': 'thr', 'to': 'out'},
        ]
        r2 = _run_batch(in_dir, out_dir, config=cfg2)
        assert r2.new_completed == 1
        assert r2.reused == 0
        assert r2.stale == 1
        assert 'pipeline config changed' in (r2.results[0].error or '')

    def test_cosmetic_config_change_is_ignored(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        _run_batch(in_dir, out_dir, config=SIMPLE_CFG)
        cfg2 = json.loads(json.dumps(SIMPLE_CFG))
        cfg2['name'] = 'a completely different name'
        cfg2['description'] = 'cosmetic only'
        r2 = _run_batch(in_dir, out_dir, config=cfg2)
        assert r2.reused == 1
        assert r2.new_completed == 0

    def test_deleted_output_forces_recompute(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png', 'b.png'))
        _run_batch(in_dir, out_dir)
        os.remove(os.path.join(out_dir, 'a_out.png'))
        r2 = _run_batch(in_dir, out_dir)
        disp = {os.path.basename(r.input_path): r for r in r2.results}
        assert disp['a.png'].disposition == 'new'
        assert disp['a.png'].stale is True
        assert disp['b.png'].disposition == 'reused'

    def test_modified_output_forces_recompute(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        _run_batch(in_dir, out_dir)
        out_p = os.path.join(out_dir, 'a_out.png')
        # Corrupt the delivered product's bytes (same path).
        write_image(alg.generate_checkerboard(8, 8, 2), out_p, fmt='PNG')
        r2 = _run_batch(in_dir, out_dir)
        assert r2.new_completed == 1
        assert r2.reused == 0
        assert r2.stale == 1
        assert 'output bytes changed' in (r2.results[0].error or '')

    def test_force_recomputes_everything(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir)
        _run_batch(in_dir, out_dir)
        r2 = _run_batch(in_dir, out_dir, force=True)
        assert r2.new_completed == 3
        assert r2.reused == 0
        assert r2.stale == 0

    def test_no_resume_computes_but_next_run_reuses(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        _run_batch(in_dir, out_dir)
        r2 = _run_batch(in_dir, out_dir, resume=False)
        assert r2.new_completed == 1 and r2.reused == 0
        r3 = _run_batch(in_dir, out_dir)
        assert r3.reused == 1


class TestMultipleOutputs:

    def test_all_planned_outputs_recorded_and_reused(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        r1 = _run_batch(in_dir, out_dir, config=COMPLEX_CFG)
        assert r1.new_completed == 1
        res = r1.results[0]
        assert len(res.planned_outputs) == 2
        assert os.path.isfile(os.path.join(out_dir, 'a_edges.png'))
        assert os.path.isfile(os.path.join(out_dir, 'a_enhanced.jpg'))
        lines = _read_ledger_lines(out_dir)
        completed = [x for x in lines if x.get('state') == STATE_COMPLETED][-1]
        assert len(completed['outputs']) == 2
        r2 = _run_batch(in_dir, out_dir, config=COMPLEX_CFG)
        assert r2.reused == 1

    def test_one_missing_output_of_many_forces_recompute(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        _run_batch(in_dir, out_dir, config=COMPLEX_CFG)
        os.remove(os.path.join(out_dir, 'a_enhanced.jpg'))
        r2 = _run_batch(in_dir, out_dir, config=COMPLEX_CFG)
        assert r2.new_completed == 1 and r2.stale == 1
        assert 'output missing' in (r2.results[0].error or '')
        assert os.path.isfile(os.path.join(out_dir, 'a_enhanced.jpg'))


# ---------------------------------------------------------------------------
# Failure, retry and invalid inputs
# ---------------------------------------------------------------------------

class TestFailuresAndRetry:

    def _failing_config(self):
        return {
            'version': '1.0', 'name': 'crop_fail',
            'nodes': [
                {'id': 'in', 'type': 'input'},
                # Crop region far larger than the tiny images -> deterministic fail.
                {'id': 'crop', 'type': 'crop',
                 'params': {'x': 0, 'y': 0, 'width': 500, 'height': 500}},
                {'id': 'out', 'type': 'output', 'params': {'suffix': '_out'}},
            ],
            'edges': [{'from': 'in', 'to': 'crop'}, {'from': 'crop', 'to': 'out'}],
        }

    def test_failure_is_persisted_and_carried_forward(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',), size=8)
        cfg = self._failing_config()
        r1 = _run_batch(in_dir, out_dir, config=cfg)
        assert r1.failed == 1 and r1.new_completed == 0
        # Second plain run carries the failure forward without recomputing.
        r2 = _run_batch(in_dir, out_dir, config=cfg)
        assert r2.failed == 1
        assert r2.results[0].disposition == 'failed'
        lines = _read_ledger_lines(out_dir)
        run2_items = [r for r in lines if r.get('type') == 'item'
                      and r.get('run_id') == r2.run_id]
        assert run2_items and run2_items[-1]['state'] == STATE_FAILED
        assert 'carried_from_run' in run2_items[-1]

    def test_retry_failed_recomputes_failed_item(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png', 'b.png'), size=8)

        # First run: a transient failure hits a.png only on its first attempt.
        from image_pipeline.utils.types import ImageProcessingResult
        executor = _make_executor(SIMPLE_CFG)
        original_run = executor.run
        state = {'a.png': True}

        def flaky_run(context):
            ip = context['input_path']
            if state.get(os.path.basename(ip)):
                state[os.path.basename(ip)] = False
                return ImageProcessingResult(input_path=ip, success=False,
                                             error='transient pipeline failure')
            return original_run(context)
        executor.run = flaky_run
        batch = BatchExecutor(executor, in_dir, out_dir, config_file='cfg.json',
                              config_raw=SIMPLE_CFG)
        r1 = batch.run()
        disp1 = {os.path.basename(r.input_path): r.disposition for r in r1.results}
        assert disp1 == {'a.png': 'failed', 'b.png': 'new'}

        # Plain rerun carries the failure forward; --retry-failed recomputes it.
        r_plain = _run_batch(in_dir, out_dir)
        assert all(r.disposition == 'failed'
                   for r in r_plain.results if os.path.basename(r.input_path) == 'a.png')
        r2 = _run_batch(in_dir, out_dir, retry_failed=True)
        disp2 = {os.path.basename(r.input_path): r.disposition for r in r2.results}
        assert disp2['a.png'] == 'new'
        assert disp2['b.png'] == 'reused'
        assert r2.failed == 0 and r2.new_completed == 1 and r2.reused == 1

    def test_retry_failed_that_fails_again_never_falls_back_to_old_output(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',), size=8)
        _run_batch(in_dir, out_dir)  # run 1: valid completion + product on disk

        from image_pipeline.utils.types import ImageProcessingResult

        def always_fail(context):
            return ImageProcessingResult(input_path=context['input_path'],
                                         success=False, error='still broken')

        # run 2: --force recompute fails; old product remains on disk.
        executor2 = _make_executor(SIMPLE_CFG)
        executor2.run = always_fail
        b2 = BatchExecutor(executor2, in_dir, out_dir, config_file='cfg.json',
                           config_raw=SIMPLE_CFG, force=True)
        r2 = b2.run()
        assert r2.results[0].disposition == 'failed'
        assert os.path.isfile(os.path.join(out_dir, 'a_out.png'))

        # run 3: --retry-failed fails again; it must NOT fall back to the
        # older verified completion.
        executor3 = _make_executor(SIMPLE_CFG)
        executor3.run = always_fail
        b3 = BatchExecutor(executor3, in_dir, out_dir, config_file='cfg.json',
                           config_raw=SIMPLE_CFG, retry_failed=True)
        r3 = b3.run()
        assert r3.results[0].disposition == 'failed'
        assert r3.new_completed == 0 and r3.reused == 0
        lines = _read_ledger_lines(out_dir)
        a_items = [x for x in lines if x.get('type') == 'item'
                   and x['input_path'].endswith('a.png')]
        assert a_items[-1]['state'] == STATE_FAILED

    def test_invalid_input_recorded_and_reevaluated(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        bad = os.path.join(in_dir, 'bad.png')
        with open(bad, 'wb') as f:
            f.write(b'garbage' * 40)
        r1 = _run_batch(in_dir, out_dir)
        assert r1.invalid == 1 and r1.failed == 1
        # Replace with a real image; next run processes it.
        write_image(alg.generate_gradient_image(8, 8), bad, fmt='PNG')
        r2 = _run_batch(in_dir, out_dir)
        assert r2.invalid == 0 and r2.new_completed == 1


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------

class TestCrashRecovery:

    def test_torn_ledger_tail_is_truncated(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        r1 = _run_batch(in_dir, out_dir)
        ledger_path = os.path.join(out_dir, '.imgpipe', 'ledger.jsonl')
        with open(ledger_path, 'ab') as f:
            f.write(b'{"type": "item", "run_id": "' + r1.run_id.encode()
                    + b'", "state": "processin')  # torn, no newline
        ledger = Ledger(out_dir)
        ledger.open()
        assert ledger.truncated_lines == 1
        # File repaired: every remaining line parses.
        lines = _read_ledger_lines(out_dir)
        assert all(isinstance(r, dict) for r in lines)
        # Re-running still works and reuses the intact completion.
        r2 = _run_batch(in_dir, out_dir)
        assert r2.reused == 1
        assert r2.ledger_truncated_lines == 0

    def test_stuck_processing_is_reconciled_and_retried(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png', 'b.png'))
        r1 = _run_batch(in_dir, out_dir)
        ledger_path = os.path.join(out_dir, '.imgpipe', 'ledger.jsonl')
        # Append a dangling PROCESSING record for a.png, simulating a kill
        # between the processing append and the completion append.
        dangling = {
            'type': 'item', 'run_id': r1.run_id, 'ts': time.time(),
            'input_path': os.path.join(in_dir, 'a.png'),
            'state': STATE_PROCESSING, 'attempt': 2,
        }
        with open(ledger_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dangling) + '\n')
        # A fresh batch run opens the ledger, reconciles the dangling item to
        # failed, and immediately retries it (interrupted items are retried
        # even without --retry-failed).
        r2 = _run_batch(in_dir, out_dir)
        disp = {os.path.basename(x.input_path): x.disposition for x in r2.results}
        assert disp['a.png'] == 'new'
        assert disp['b.png'] == 'reused'
        assert r2.recovered_processing == 1
        # The reconciled FAILED transition is persisted in the ledger.
        lines = _read_ledger_lines(out_dir)
        reconciled = [x for x in lines if x.get('recovered_processing')]
        assert reconciled and reconciled[-1]['state'] == STATE_FAILED

    def test_leftover_temp_files_are_swept(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(out_dir, exist_ok=True)
        _make_inputs(in_dir, names=('a.png',))
        leftover = os.path.join(out_dir, TEMP_FILE_PREFIX + 'abc123.png')
        with open(leftover, 'wb') as f:
            f.write(b'partial image data')
        r = _run_batch(in_dir, out_dir)
        assert not os.path.exists(leftover)
        assert r.temp_files_removed == 1

    def test_no_completed_record_without_valid_product(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        _run_batch(in_dir, out_dir)
        lines = _read_ledger_lines(out_dir)
        completed = [x for x in lines if x.get('state') == STATE_COMPLETED]
        assert len(completed) == 1
        rec = completed[-1]
        # Every recorded output exists and matches the fingerprint.
        for o in rec['outputs']:
            assert os.path.isfile(o['path'])
            assert file_sha256(o['path']) == o['sha256']

    def test_completion_appended_after_product_lands(self, tmpdir_path, monkeypatch):
        # Simulate a crash: output write completes to a temp file but the
        # process dies before os.replace / ledger append.  Nothing may be
        # marked completed.
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        executor = _make_executor()
        batch = BatchExecutor(executor, in_dir, out_dir, config_file='cfg.json',
                              config_raw=SIMPLE_CFG)
        import image_pipeline.batch.executor as exec_mod
        # Force failure at the post-run fingerprint step.
        orig = exec_mod.describe_output_file

        def boom(path):
            raise OSError('simulated disk failure during fingerprint')
        monkeypatch.setattr(exec_mod, 'describe_output_file', boom)
        report = batch.run()
        assert report.failed == 1
        assert report.new_completed == 0
        lines = _read_ledger_lines(out_dir)
        assert not [x for x in lines if x.get('state') == STATE_COMPLETED]
        monkeypatch.setattr(exec_mod, 'describe_output_file', orig)
        # Next run with --retry-failed recovers and completes normally.
        r2 = _run_batch(in_dir, out_dir, retry_failed=True)
        assert r2.new_completed == 1

    def test_resume_mid_batch_continues_remaining(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir)
        executor = _make_executor()
        call_count = {'n': 0}
        original_run = executor.run

        def run_then_die(context):
            call_count['n'] += 1
            result = original_run(context)
            if call_count['n'] == 2:
                # Die right after the second image's products hit disk but
                # before its COMPLETED ledger line: emulate by deleting the
                # ledger's pending completion is hard; instead kill between
                # processing and completion via raising here only when the
                # output already exists.
                out_p = os.path.join(out_dir, 'b_out.png')
                if os.path.isfile(out_p):
                    raise KeyboardInterrupt()
            return result
        executor.run = run_then_die
        batch = BatchExecutor(executor, in_dir, out_dir, config_file='cfg.json',
                              config_raw=SIMPLE_CFG)
        with pytest.raises(KeyboardInterrupt):
            batch.run()
        # First image completed; second's product may exist but ledger shows
        # it processing -> must be retried; third never touched.
        r2 = _run_batch(in_dir, out_dir)
        disp = {os.path.basename(x.input_path): x.disposition for x in r2.results}
        assert disp['a.png'] == 'reused'
        assert disp['b.png'] == 'new'
        assert disp['c.png'] == 'new'
        assert r2.reused == 1 and r2.new_completed == 2


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

class TestFingerprint:

    def test_fingerprint_stable_for_same_graph(self):
        c1 = json.loads(json.dumps(SIMPLE_CFG))
        c2 = json.loads(json.dumps(SIMPLE_CFG))
        assert config_fingerprint(c1)['hash'] == config_fingerprint(c2)['hash']

    def test_fingerprint_changes_with_params(self):
        c1 = json.loads(json.dumps(SIMPLE_CFG))
        c2 = json.loads(json.dumps(SIMPLE_CFG))
        c2['nodes'][2]['params'] = {'suffix': '_different'}
        assert config_fingerprint(c1)['hash'] != config_fingerprint(c2)['hash']

    def test_ledger_records_full_config_content(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        _run_batch(in_dir, out_dir)
        run_line = [r for r in _read_ledger_lines(out_dir) if r['type'] == 'run'][0]
        assert run_line['config_hash']
        assert run_line['config']['nodes'] == SIMPLE_CFG['nodes']
        assert run_line['input_dir'].endswith('in')
