import json
import os

import pytest

from image_pipeline.algorithms import core as alg
from image_pipeline.batch.executor import BatchExecutor
from image_pipeline.batch import ledger as ledger_mod
from image_pipeline.batch.ledger import (
    Ledger, AtomicImageWriter, cleanup_temp_files,
    config_hash, default_ledger_path, is_temp_file_name,
    REASON_CONFIG_CHANGED, REASON_INPUT_CHANGED, REASON_OUTPUT_MISSING,
    REASON_OUTPUT_CHANGED, REASON_OUTPUT_PLAN_CHANGED,
)
from image_pipeline.config.loader import PipelineConfig, load_config_file
from image_pipeline.utils.image_io import write_image, read_image
from image_pipeline.utils.sample_generator import generate_all


def _config():
    return {
        'version': '1.0', 'name': 'test_pipeline',
        'nodes': [
            {'id': 'in', 'type': 'input'},
            {'id': 'gray', 'type': 'grayscale'},
            {'id': 'thresh', 'type': 'threshold', 'params': {'value': 128}},
            {'id': 'out', 'type': 'output', 'params': {'suffix': '_out'}},
        ],
        'edges': [
            {'from': 'in', 'to': 'gray'},
            {'from': 'gray', 'to': 'thresh'},
            {'from': 'thresh', 'to': 'out'},
        ],
    }


def _make_executor(config=None):
    cfg = PipelineConfig(config or _config())
    executor, validation = cfg.build_executor()
    assert validation.valid, validation.errors
    return executor


def _make_inputs(in_dir, names=('a.png', 'b.png', 'c.png')):
    os.makedirs(in_dir, exist_ok=True)
    for i, name in enumerate(names):
        write_image(alg.generate_gradient_image(8 + i, 8 + i),
                    os.path.join(in_dir, name), fmt='PNG')


def _run(executor, in_dir, out_dir, config=None, **kw):
    batch = BatchExecutor(executor, in_dir, out_dir,
                          config_raw=config or _config(), **kw)
    return batch.run()


@pytest.fixture
def workspace(tmpdir_path):
    in_dir = os.path.join(tmpdir_path, 'in')
    out_dir = os.path.join(tmpdir_path, 'out')
    _make_inputs(in_dir)
    return in_dir, out_dir


class TestFreshRunWithLedger:

    def test_first_run_completes_all_and_writes_ledger(self, workspace):
        in_dir, out_dir = workspace
        report = _run(_make_executor(), in_dir, out_dir)
        assert report.newly_completed == 3
        assert report.reused_count == 0
        assert report.failed == 0
        assert report.succeeded == 3
        assert os.path.isfile(default_ledger_path(out_dir))
        for name in ('a_out.png', 'b_out.png', 'c_out.png'):
            assert os.path.isfile(os.path.join(out_dir, name))

    def test_ledger_records_config_input_plan_and_transitions(self, workspace):
        in_dir, out_dir = workspace
        _run(_make_executor(), in_dir, out_dir)
        events = [json.loads(line) for line in
                  open(default_ledger_path(out_dir), encoding='utf-8')]
        kinds = [e['t'] for e in events]
        assert kinds.count('run_start') == 1
        assert kinds.count('plan') == 3
        # processing -> completed for each image
        assert kinds.count('state') == 6
        start = events[0]
        assert start['config']['name'] == 'test_pipeline'
        assert start['config_hash'] == config_hash(_config())
        plans = [e for e in events if e['t'] == 'plan']
        assert plans[0]['input']['sha256']
        assert plans[0]['input']['size'] > 0
        assert plans[0]['outputs'][0]['path'] == 'a_out.png'
        completed = [e for e in events if e.get('state') == 'completed']
        assert {e['item'] for e in completed} == {'a.png', 'b.png', 'c.png'}
        for e in completed:
            assert e['outputs'][0]['sha256']
            assert e['outputs'][0]['size'] > 0
            assert e['input_sha256']

    def test_ordering_is_processing_then_completed(self, workspace):
        in_dir, out_dir = workspace
        _run(_make_executor(), in_dir, out_dir)
        events = [json.loads(line) for line in
                  open(default_ledger_path(out_dir), encoding='utf-8')]
        states = [(e['item'], e.get('state')) for e in events if e['t'] == 'state']
        # Each image's last two states are processing, completed; completed
        # never appears before its artifact write.
        for i in range(0, len(states), 2):
            assert states[i][1] == 'processing'
            assert states[i + 1][1] == 'completed'
            assert states[i][0] == states[i + 1][0]


class TestResumeAndReuse:

    def test_second_run_reuses_everything(self, workspace):
        in_dir, out_dir = workspace
        exec1 = _make_executor()
        r1 = _run(exec1, in_dir, out_dir)
        mtimes = {n: os.path.getmtime(os.path.join(out_dir, n))
                  for n in ('a_out.png', 'b_out.png', 'c_out.png')}
        r2 = _run(_make_executor(), in_dir, out_dir)
        assert r2.newly_completed == 0
        assert r2.reused_count == 3
        assert r2.skipped == 3
        assert r2.failed == 0
        # Artifacts untouched
        for n, mt in mtimes.items():
            assert os.path.getmtime(os.path.join(out_dir, n)) == mt
        reused = [r for r in r2.results if r.disposition == 'reused']
        assert len(reused) == 3
        assert reused[0].reused_from_run == r1.run_id
        assert reused[0].attempted is False

    def test_reuse_skipped_items_counted_in_report_dict(self, workspace):
        in_dir, out_dir = workspace
        _run(_make_executor(), in_dir, out_dir)
        r2 = _run(_make_executor(), in_dir, out_dir)
        d = r2.to_dict()
        assert d['summary']['newly_completed'] == 0
        assert d['summary']['reused'] == 3
        assert d['run']['run_id'] == r2.run_id
        assert d['results'][0]['disposition'] == 'reused'

    def test_resume_after_partial_run(self, tmpdir_path):
        # Simulate a run that only completed the first image, then resume.
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir)
        executor = _make_executor()
        seen = []

        def cb(done, total, result):
            seen.append(result.input_path)

        r1 = _run(executor, in_dir, out_dir, progress_callback=cb)
        assert r1.newly_completed == 3
        # Remove two outputs and truncate their ledger records to simulate
        # that they never completed: rebuild ledger from scratch with one item.
        os.remove(os.path.join(out_dir, 'b_out.png'))
        os.remove(os.path.join(out_dir, 'c_out.png'))
        ledger_path = default_ledger_path(out_dir)
        lines = open(ledger_path, encoding='utf-8').readlines()
        keep = [ln for ln in lines
                if not json.loads(ln).get('item', '').startswith(('b', 'c'))]
        with open(ledger_path, 'w', encoding='utf-8') as f:
            f.writelines(keep)
        r2 = _run(_make_executor(), in_dir, out_dir)
        assert r2.reused_count == 1
        assert r2.newly_completed == 2
        assert os.path.isfile(os.path.join(out_dir, 'b_out.png'))
        assert os.path.isfile(os.path.join(out_dir, 'c_out.png'))


class TestInvalidation:

    def _second_config(self, value=200):
        cfg = _config()
        cfg['nodes'][2]['params']['value'] = value
        return cfg

    def test_changed_config_invalidates_and_recomputes(self, workspace):
        in_dir, out_dir = workspace
        _run(_make_executor(), in_dir, out_dir)
        cfg2 = self._second_config(200)
        r2 = _run(_make_executor(cfg2), in_dir, out_dir, config=cfg2)
        assert r2.newly_completed == 3
        assert r2.invalidated_count == 3
        redone = [r for r in r2.results if r.invalidation_reason == REASON_CONFIG_CHANGED]
        assert len(redone) == 3

    def test_changed_input_bytes_invalidates(self, workspace):
        in_dir, out_dir = workspace
        _run(_make_executor(), in_dir, out_dir)
        # Replace a.png content under the same name
        write_image(alg.generate_checkerboard(9, 9, 3),
                    os.path.join(in_dir, 'a.png'), fmt='PNG')
        r2 = _run(_make_executor(), in_dir, out_dir)
        assert r2.newly_completed == 1
        assert r2.reused_count == 2
        item = [r for r in r2.results if r.input_path.endswith('a.png')][0]
        assert item.invalidation_reason == REASON_INPUT_CHANGED

    def test_deleted_output_invalidates(self, workspace):
        in_dir, out_dir = workspace
        _run(_make_executor(), in_dir, out_dir)
        os.remove(os.path.join(out_dir, 'a_out.png'))
        r2 = _run(_make_executor(), in_dir, out_dir)
        assert r2.newly_completed == 1
        assert r2.reused_count == 2
        item = [r for r in r2.results if r.input_path.endswith('a.png')][0]
        assert item.invalidation_reason == REASON_OUTPUT_MISSING

    def test_modified_output_invalidates(self, workspace):
        in_dir, out_dir = workspace
        _run(_make_executor(), in_dir, out_dir)
        p = os.path.join(out_dir, 'a_out.png')
        data = open(p, 'rb').read()
        with open(p, 'wb') as f:
            f.write(data + b' ')
        r2 = _run(_make_executor(), in_dir, out_dir)
        assert r2.newly_completed == 1
        assert r2.reused_count == 2
        item = [r for r in r2.results if r.input_path.endswith('a.png')][0]
        assert item.invalidation_reason in (REASON_OUTPUT_CHANGED, REASON_OUTPUT_MISSING)

    def test_changed_output_plan_invalidates(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        _run(_make_executor(), in_dir, out_dir)
        cfg2 = _config()
        cfg2['nodes'][3]['params']['suffix'] = '_processed'
        r2 = _run(_make_executor(cfg2), in_dir, out_dir, config=cfg2)
        assert r2.newly_completed == 1
        item = r2.results[0]
        # The suffix lives in the config, so the config hash is the first
        # mismatch detected (the planned-path check is a defensive backstop).
        assert item.invalidation_reason in (
            REASON_CONFIG_CHANGED, REASON_OUTPUT_PLAN_CHANGED)
        assert os.path.isfile(os.path.join(out_dir, 'a_processed.png'))

    def test_force_recompute_ignores_good_records(self, workspace):
        in_dir, out_dir = workspace
        _run(_make_executor(), in_dir, out_dir)
        mtimes = {n: os.path.getmtime(os.path.join(out_dir, n))
                  for n in ('a_out.png', 'b_out.png', 'c_out.png')}
        import time as _t
        _t.sleep(0.02)
        r2 = _run(_make_executor(), in_dir, out_dir, force_recompute=True)
        assert r2.newly_completed == 3
        assert r2.reused_count == 0
        assert r2.forced_count == 3
        d = r2.to_dict()
        assert d['summary']['forced'] == 3
        assert d['results'][0]['forced'] is True
        for n, mt in mtimes.items():
            assert os.path.getmtime(os.path.join(out_dir, n)) > mt


class TestFailures:

    def _mixed_dir(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in_mixed')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8),
                    os.path.join(in_dir, 'good.png'), fmt='PNG')
        with open(os.path.join(in_dir, 'bad.png'), 'wb') as f:
            f.write(b'not a real png' * 10)
        return in_dir, out_dir

    def test_failure_recorded_and_not_retried_by_default(self, tmpdir_path):
        in_dir, out_dir = self._mixed_dir(tmpdir_path)
        r1 = _run(_make_executor(), in_dir, out_dir)
        assert r1.failed == 1
        assert r1.newly_completed == 1
        r2 = _run(_make_executor(), in_dir, out_dir)
        assert r2.failed == 1
        assert r2.newly_completed == 0
        assert r2.reused_count == 1
        bad = [r for r in r2.results if r.input_path.endswith('bad.png')][0]
        assert bad.attempted is False
        assert bad.disposition == 'failed'

    def test_retry_failed_flag_retries_failure(self, tmpdir_path):
        in_dir, out_dir = self._mixed_dir(tmpdir_path)
        _run(_make_executor(), in_dir, out_dir)
        # Repair the bad input; --retry-failed should pick it up.
        write_image(alg.generate_gradient_image(8, 8),
                    os.path.join(in_dir, 'bad.png'), fmt='PNG')
        r2 = _run(_make_executor(), in_dir, out_dir, retry_failed=True)
        assert r2.failed == 0
        assert r2.reused_count == 1
        assert r2.newly_completed == 1

    def test_changed_input_auto_retries_stale_failure(self, tmpdir_path):
        in_dir, out_dir = self._mixed_dir(tmpdir_path)
        _run(_make_executor(), in_dir, out_dir)
        write_image(alg.generate_gradient_image(8, 8),
                    os.path.join(in_dir, 'bad.png'), fmt='PNG')
        # Without --retry-failed the stale failure is retried because the
        # input bytes no longer match the failed record.
        r2 = _run(_make_executor(), in_dir, out_dir)
        assert r2.failed == 0
        assert r2.newly_completed == 1


class TestCrashRecovery:

    def test_stale_processing_is_retried(self, workspace):
        in_dir, out_dir = workspace
        _run(_make_executor(), in_dir, out_dir)
        ledger_path = default_ledger_path(out_dir)
        # Append a dangling processing event for a.png (crash mid-image).
        with open(ledger_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'t': 'state', 'run_id': 'oldrun',
                                'item': 'a.png', 'state': 'processing'}) + '\n')
        led = Ledger.open(out_dir)
        assert led.recovered_interrupted >= 1
        led.close()
        r2 = _run(_make_executor(), in_dir, out_dir)
        # a.png is redone (its outputs are re-validated but processing forces
        # a retry); b/c reused.
        assert r2.newly_completed == 1
        assert r2.reused_count == 2
        assert r2.recovered_interrupted >= 1

    def test_corrupt_tail_is_truncated_and_run_continues(self, workspace):
        in_dir, out_dir = workspace
        _run(_make_executor(), in_dir, out_dir)
        ledger_path = default_ledger_path(out_dir)
        good_size = os.path.getsize(ledger_path)
        with open(ledger_path, 'ab') as f:
            f.write(b'{"t":"state","state":"processin')  # torn partial line
        r2 = _run(_make_executor(), in_dir, out_dir)
        assert r2.ledger_repaired_tail is True
        assert r2.reused_count == 3
        # Ledger must be left parseable, with the garbage gone.
        for line in open(ledger_path, encoding='utf-8'):
            json.loads(line)

    def test_corrupt_middle_line_keeps_prefix(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        _run(_make_executor(), in_dir, out_dir)
        ledger_path = default_ledger_path(out_dir)
        content = open(ledger_path, 'rb').read()
        # Insert garbage followed by more "valid-looking" events; recovery may
        # only trust the prefix and must produce a parseable ledger.
        with open(ledger_path, 'ab') as f:
            f.write(b'\xff\xfe not json\n')
            f.write(json.dumps({'t': 'plan', 'item': 'ghost'}).encode())
        led = Ledger.open(out_dir)
        led.close()
        for line in open(ledger_path, encoding='utf-8'):
            json.loads(line)

    def test_temp_files_cleaned_on_open(self, workspace):
        in_dir, out_dir = workspace
        os.makedirs(out_dir, exist_ok=True)
        leftover = os.path.join(out_dir, '.a_out.png.deadbeef.imgpipe.tmp')
        with open(leftover, 'wb') as f:
            f.write(b'partial')
        other = os.path.join(out_dir, 'real_output.png')
        with open(other, 'wb') as f:
            f.write(b'keep')
        led = Ledger.open(out_dir)
        led.close()
        assert not os.path.exists(leftover)
        assert os.path.exists(other)

    def test_atomic_writer_commits_via_rename(self, tmpdir_path):
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(out_dir, exist_ok=True)
        writer = AtomicImageWriter('testrun')
        target = os.path.join(out_dir, 'x.png')
        writer(alg.generate_gradient_image(4, 4), target, fmt='PNG')
        assert os.path.isfile(target)
        leftovers = [n for n in os.listdir(out_dir) if is_temp_file_name(n)]
        assert leftovers == []
        assert read_image(target) is not None

    def test_atomic_writer_removes_temp_on_failure(self, tmpdir_path, monkeypatch):
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(out_dir, exist_ok=True)
        writer = AtomicImageWriter('testrun')
        from PIL import Image as PILImage
        def boom(*a, **k):
            raise OSError('disk full')
        monkeypatch.setattr(PILImage.Image, 'save', boom)
        target = os.path.join(out_dir, 'x.png')
        with pytest.raises(OSError):
            writer(alg.generate_gradient_image(4, 4), target, fmt='PNG')
        assert not os.path.exists(target)
        leftovers = [n for n in os.listdir(out_dir) if is_temp_file_name(n)]
        assert leftovers == []

    def test_no_completed_without_artifact(self, tmpdir_path, monkeypatch):
        # If the write fails, the ledger must never claim completed.
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        from PIL import Image as PILImage
        calls = {'n': 0}
        real_save = PILImage.Image.save

        def fail_save(self, fp, *a, **k):
            calls['n'] += 1
            raise OSError('simulated write failure')

        monkeypatch.setattr(PILImage.Image, 'save', fail_save)
        report = _run(_make_executor(), in_dir, out_dir)
        assert report.failed == 1
        assert report.newly_completed == 0
        events = [json.loads(l) for l in
                  open(default_ledger_path(out_dir), encoding='utf-8')]
        assert not any(e.get('state') == 'completed' for e in events)
        assert any(e.get('state') == 'failed' for e in events)
        # No partial target left behind
        assert not os.path.exists(os.path.join(out_dir, 'a_out.png'))
        assert not [n for n in os.listdir(out_dir) if is_temp_file_name(n)]


class TestStaleRecords:

    def test_vanished_input_reported_as_stale(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png', 'b.png'))
        _run(_make_executor(), in_dir, out_dir)
        os.remove(os.path.join(in_dir, 'a.png'))
        r2 = _run(_make_executor(), in_dir, out_dir)
        assert r2.stale_count == 1
        assert r2.total == 1
        stale = [r for r in r2.results if r.disposition == 'stale'][0]
        assert stale.attempted is False
        assert 'a.png' in stale.input_path


class TestNoLedger:

    def test_no_ledger_processes_every_time(self, workspace):
        in_dir, out_dir = workspace
        r1 = _run(_make_executor(), in_dir, out_dir, use_ledger=False)
        r2 = _run(_make_executor(), in_dir, out_dir, use_ledger=False)
        assert r1.newly_completed == 3
        assert r2.newly_completed == 3
        assert r2.reused_count == 0
        assert not os.path.exists(os.path.join(out_dir, '.imgpipe'))


class TestMultipleOutputs:

    def _branch_config(self):
        return {
            'version': '1.0', 'name': 'branch',
            'nodes': [
                {'id': 'in', 'type': 'input'},
                {'id': 'gray', 'type': 'grayscale'},
                {'id': 'o1', 'type': 'output', 'params': {'suffix': '_g'}},
                {'id': 'o2', 'type': 'output', 'params': {'suffix': '_e'}},
                {'id': 'sob', 'type': 'sobel', 'params': {'direction': 'x'}},
            ],
            'edges': [
                {'from': 'in', 'to': 'gray'},
                {'from': 'gray', 'to': 'o1'},
                {'from': 'gray', 'to': 'sob'},
                {'from': 'sob', 'to': 'o2'},
            ],
        }

    def test_all_outputs_committed_and_reused_then_invalidated(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _make_inputs(in_dir, names=('a.png',))
        cfg = self._branch_config()
        r1 = _run(_make_executor(cfg), in_dir, out_dir, config=cfg)
        assert r1.newly_completed == 1
        assert os.path.isfile(os.path.join(out_dir, 'a_g.png'))
        assert os.path.isfile(os.path.join(out_dir, 'a_e.png'))
        r2 = _run(_make_executor(cfg), in_dir, out_dir, config=cfg)
        assert r2.reused_count == 1
        # Remove one of two outputs: item must be redone
        os.remove(os.path.join(out_dir, 'a_e.png'))
        r3 = _run(_make_executor(cfg), in_dir, out_dir, config=cfg)
        assert r3.newly_completed == 1
        assert r3.results[0].invalidation_reason == REASON_OUTPUT_MISSING
