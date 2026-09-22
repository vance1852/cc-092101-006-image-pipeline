from typing import Any, Dict, List, Optional, Callable
import os
import json
import time
import traceback
from ..pipeline.engine import PipelineExecutor
from ..nodes.definitions import OutputNode, resolve_output_path
from ..utils.types import BatchReport, ImageProcessingResult, ValidationError
from ..utils.image_io import find_images, is_valid_image
from .ledger import (
    Ledger, AtomicImageWriter,
    config_hash as _config_hash,
    sha256_file,
    STATE_COMPLETED,
    STATE_FAILED,
)
from .. import __version__


def graph_config_signature(executor: PipelineExecutor) -> Dict[str, Any]:
    """Deterministic configuration description built from a live graph.

    Used when the BatchExecutor is constructed without the raw config dict;
    it captures every relevant detail (node type, effective params, edges).
    """
    nodes = []
    for nid in executor.execution_order:
        node = executor.graph.nodes[nid]
        ntype = node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type)
        nodes.append({'id': nid, 'type': ntype, 'params': node.effective_params()})
    edges = []
    for tgt, srcs in executor.graph.upstream.items():
        for src, slot in srcs:
            edges.append({'from': src, 'to': tgt, 'slot': slot})
    return {'nodes': nodes, 'edges': sorted(edges, key=lambda e: (e['from'], e['to'], e['slot']))}


class BatchExecutor:

    def __init__(self, pipeline_executor: PipelineExecutor, input_dir: str, output_dir: str,
                 config_file: str = '', progress_callback: Optional[Callable] = None,
                 config_raw: Optional[Dict[str, Any]] = None,
                 use_ledger: bool = True, ledger_path: Optional[str] = None,
                 retry_failed: bool = False, force_recompute: bool = False):
        self.executor = pipeline_executor
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config_file = config_file
        self.progress_callback = progress_callback
        self.config_raw = config_raw if config_raw is not None else graph_config_signature(pipeline_executor)
        self.use_ledger = use_ledger
        self.ledger_path = ledger_path
        self.retry_failed = retry_failed
        self.force_recompute = force_recompute
        self._cfg_hash = _config_hash(self.config_raw)

    def _ensure_output_dir(self) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def _collect_input_images(self) -> List[str]:
        if not os.path.isdir(self.input_dir):
            raise ValidationError(f"Input directory does not exist: '{self.input_dir}'")
        return find_images(self.input_dir)

    # ------------------------------------------------------------ planning

    def _output_nodes(self) -> List[OutputNode]:
        return self.executor.graph.get_output_nodes()

    def _plan_outputs(self, filename: str) -> List[Dict[str, str]]:
        planned = []
        for node in self._output_nodes():
            abs_path = resolve_output_path(filename, self.output_dir, node.effective_params())
            planned.append({'node': node.node_id,
                            'path': os.path.relpath(abs_path, self.output_dir)})
        return planned

    def _input_identity(self, img_path: str) -> Dict[str, Any]:
        identity: Dict[str, Any] = {'name': os.path.basename(img_path)}
        try:
            identity['size'] = os.path.getsize(img_path)
            identity['sha256'] = sha256_file(img_path)
        except OSError as e:
            identity['size'] = None
            identity['sha256'] = None
            identity['error'] = f'cannot stat/hash input: {e}'
        return identity

    def _output_identities(self, rel_paths: List[str]) -> List[Dict[str, Any]]:
        identities = []
        for rel in rel_paths:
            abs_path = os.path.join(self.output_dir, rel)
            identities.append({
                'path': rel,
                'size': os.path.getsize(abs_path),
                'sha256': sha256_file(abs_path),
            })
        return identities

    # ------------------------------------------------------------ run

    def run(self) -> BatchReport:
        report = BatchReport(pipeline_config_file=self.config_file, input_dir=self.input_dir, output_dir=self.output_dir)
        report.mode = ('force_recompute' if self.force_recompute
                       else 'retry_failed' if self.retry_failed else 'normal')
        overall_start = time.perf_counter()
        try:
            self._ensure_output_dir()
        except Exception as e:
            self._finish_timing(report, overall_start)
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False,
                                          disposition='failed',
                                          error=f'Failed to create output directory: {e}')
            report.total = report.succeeded = report.failed = 0
            report.results.append(dummy)
            return report
        try:
            image_paths = self._collect_input_images()
        except ValidationError as e:
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False,
                                          disposition='failed', error=str(e))
            report.results.append(dummy)
            self._finish_timing(report, overall_start)
            return report

        ledger: Optional[Ledger] = None
        if self.use_ledger:
            ledger = Ledger.open(self.output_dir, self.ledger_path)
            ledger.run_started(
                self.config_raw, self._cfg_hash, self.input_dir, self.output_dir,
                self.config_file, __version__, report.mode)
            report.ledger_path = ledger.path
            report.run_id = ledger.run_id
            report.recovered_interrupted = ledger.recovered_interrupted
            report.temp_files_cleaned = len(ledger.temp_files_removed)
            report.ledger_repaired_tail = ledger.truncated_corrupt_tail

        report.total = len(image_paths)
        current_keys = set()

        try:
            for idx, img_path in enumerate(image_paths):
                filename = os.path.basename(img_path)
                item_key = filename
                current_keys.add(item_key)
                planned_outputs = self._plan_outputs(filename)
                input_identity = self._input_identity(img_path)

                if ledger is not None:
                    ledger.item_planned(item_key, input_identity, planned_outputs)

                img_result = self._process_one(
                    idx, img_path, filename, item_key, planned_outputs,
                    input_identity, ledger)

                if img_result.disposition == 'reused':
                    report.reused_count += 1
                    report.skipped += 1
                    report.succeeded += 1
                elif img_result.success:
                    report.newly_completed += 1
                    report.succeeded += 1
                    if img_result.forced:
                        report.forced_count += 1
                    elif img_result.invalidation_reason:
                        report.invalidated_count += 1
                else:
                    report.failed += 1
                    if img_result.forced:
                        report.forced_count += 1
                    elif img_result.invalidation_reason:
                        report.invalidated_count += 1
                report.results.append(img_result)

                if self.progress_callback:
                    try:
                        self.progress_callback(idx + 1, report.total, img_result)
                    except Exception:
                        pass

            # Records from the previous run whose inputs are no longer part of
            # the current input set: stale ledger entries, reported but not run.
            if ledger is not None:
                for key in ledger.orphan_keys(current_keys):
                    rec = ledger.prior_record(key)
                    stale = ImageProcessingResult(
                        input_path=os.path.join(self.input_dir, key),
                        success=(rec is not None and rec.last_state == STATE_COMPLETED),
                        disposition='stale', attempted=False,
                        error='input no longer present in this run')
                    report.stale_count += 1
                    report.results.append(stale)

            if ledger is not None:
                ledger.run_finished()
        finally:
            if ledger is not None:
                ledger.close()

        self._finish_timing(report, overall_start)
        return report

    def _process_one(self, idx: int, img_path: str, filename: str, item_key: str,
                     planned_outputs: List[Dict[str, str]],
                     input_identity: Dict[str, Any],
                     ledger: Optional[Ledger]) -> ImageProcessingResult:
        input_sha = input_identity.get('sha256') or ''

        # ---------------------------------------------------------- decide
        invalidation_reason: Optional[str] = None
        attempt = True
        reused_outputs: List[Dict[str, Any]] = []
        reused_from_run: Optional[str] = None
        forced = False

        if ledger is not None:
            prior = ledger.prior_record(item_key)
            forced = bool(self.force_recompute and prior is not None)
            if self.force_recompute:
                pass  # every item is reprocessed; prior records ignored
            elif prior is not None and prior.last_state == STATE_COMPLETED:
                reason = ledger.reuse_invalidation(
                    item_key, input_sha, planned_outputs,
                    self._cfg_hash, self.output_dir)
                if reason is None:
                    attempt = False
                    reused_outputs = ledger.completed_outputs(item_key)
                    reused_from_run = prior.completed.get('run_id')
                else:
                    # A completed record that no longer proves a valid product.
                    invalidation_reason = reason
            elif prior is not None and prior.last_state == STATE_FAILED \
                    and not self.retry_failed:
                # Reuse the recorded failure only if it was produced with the
                # same input bytes and configuration; otherwise it is stale and
                # the item is retried automatically.
                stale = ledger.failure_staleness_reason(item_key, input_sha, self._cfg_hash)
                if stale is None:
                    return ImageProcessingResult(
                        input_path=img_path,
                        output_path=self._first_output_abspath(prior),
                        success=False, disposition='failed', attempted=False,
                        error=prior.last_error or 'failed in a previous run (not retried)')
                # Stale failure (config or input changed): fall through, retry.
                invalidation_reason = stale
            # prior == None or last_state == 'processing' (interrupted):
            # always fall through to a fresh attempt.
        else:
            forced = False

        if not attempt:
            abs_outputs = [os.path.join(self.output_dir, o['path']) for o in reused_outputs]
            return ImageProcessingResult(
                input_path=img_path,
                output_path=abs_outputs[0] if abs_outputs else None,
                output_paths=abs_outputs,
                success=True, disposition='reused', attempted=False,
                reused_from_run=reused_from_run)

        # ---------------------------------------------------------- attempt
        if ledger is not None:
            ledger.mark_processing(item_key)

        if input_identity.get('sha256') is None:
            error = f'Cannot read input file: {input_identity.get("error", "unreadable")}'
            if ledger is not None:
                ledger.mark_failed(item_key, error, self._cfg_hash, input_sha or None)
            return ImageProcessingResult(
                input_path=img_path, success=False,
                disposition='failed', invalidation_reason=invalidation_reason,
                forced=forced, error=error)

        if not is_valid_image(img_path):
            error = 'Image failed pre-check verification (likely corrupt or unsupported format)'
            if ledger is not None:
                ledger.mark_failed(item_key, error, self._cfg_hash, input_sha)
            return ImageProcessingResult(
                input_path=img_path, success=False,
                disposition='failed', invalidation_reason=invalidation_reason,
                forced=forced, error=error)

        context: Dict[str, Any] = {
            'input_path': img_path,
            'input_filename': filename,
            'output_dir': self.output_dir,
            'image_index': idx,
            '_written_outputs': [],
        }
        if ledger is not None:
            context['image_writer'] = AtomicImageWriter(ledger.run_id)

        start = time.perf_counter()
        try:
            img_result = self.executor.run(context)
        except Exception as e:
            img_result = ImageProcessingResult(
                input_path=img_path, success=False,
                error=f'Unexpected error during execution: {e}\n{traceback.format_exc()}')

        img_result.duration_ms = (time.perf_counter() - start) * 1000.0
        written = context.get('_written_outputs') or []
        img_result.output_paths = list(written)
        if not img_result.output_path and written:
            img_result.output_path = written[0]
        img_result.invalidation_reason = invalidation_reason
        img_result.forced = forced

        if ledger is not None:
            if img_result.success:
                expected_rel = [o['path'] for o in planned_outputs]
                actual_rel = sorted(os.path.relpath(p, self.output_dir) for p in written)
                # The artifact set committed to disk must match the plan.
                if actual_rel != sorted(expected_rel):
                    img_result.success = False
                    img_result.error = (f'Committed outputs {actual_rel} do not match '
                                        f'planned outputs {sorted(expected_rel)}')
                    ledger.mark_failed(item_key, img_result.error, self._cfg_hash, input_sha)
                else:
                    try:
                        identities = self._output_identities(expected_rel)
                    except OSError as e:
                        img_result.success = False
                        img_result.error = f'Committed output unreadable after write: {e}'
                        ledger.mark_failed(item_key, img_result.error, self._cfg_hash, input_sha)
                    else:
                        # All outputs durable and hashed: only now record success.
                        ledger.mark_completed(item_key, self._cfg_hash, identities, input_sha)
            else:
                ledger.mark_failed(item_key, img_result.error or 'execution failed',
                                   self._cfg_hash, input_sha)

        img_result.disposition = 'completed' if img_result.success else 'failed'
        return img_result

    def _first_output_abspath(self, rec) -> Optional[str]:
        if rec is not None and rec.completed is not None:
            outs = rec.completed.get('outputs', [])
            if outs:
                return os.path.join(self.output_dir, outs[0]['path'])
        return None

    @staticmethod
    def _finish_timing(report: BatchReport, overall_start: float) -> None:
        report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0

    def write_report(self, report: BatchReport, path: str = None) -> str:
        if path is None:
            path = os.path.join(self.output_dir, 'batch_report.json')
        report_dir = os.path.dirname(path)
        if report_dir and (not os.path.exists(report_dir)):
            os.makedirs(report_dir, exist_ok=True)
        tmp_path = path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        return path


def print_text_report(report: BatchReport, verbose: bool = False) -> str:
    lines = []
    lines.append('=' * 60)
    lines.append('BATCH PROCESSING REPORT')
    lines.append('=' * 60)
    lines.append(f'Pipeline config : {report.pipeline_config_file}')
    lines.append(f'Input directory : {report.input_dir}')
    lines.append(f'Output directory: {report.output_dir}')
    if report.ledger_path:
        lines.append(f'Ledger          : {report.ledger_path}')
        lines.append(f'Run ID          : {report.run_id}  (mode: {report.mode})')
    lines.append('')
    lines.append('--- Summary ---')
    lines.append(f'Total images    : {report.total}')
    lines.append(f'Succeeded       : {report.succeeded}')
    lines.append(f'  newly done    : {report.newly_completed}')
    lines.append(f'  reused        : {report.reused_count}')
    lines.append(f'Failed          : {report.failed}')
    lines.append(f'Invalidated     : {report.invalidated_count}')
    lines.append(f'Forced recompute: {report.forced_count}')
    lines.append(f'Stale records   : {report.stale_count}')
    lines.append(f'Skipped         : {report.skipped}')
    if report.recovered_interrupted:
        lines.append(f'Recovered (was processing): {report.recovered_interrupted}')
    lines.append(f'Total duration  : {report.total_duration_ms:.2f} ms')
    if report.total > 0:
        lines.append(f'Avg per image   : {report.total_duration_ms / report.total:.2f} ms')
    lines.append('')
    if verbose:
        lines.append('--- Per-Image Details ---')
        for r in report.results:
            if r.disposition == 'reused':
                status = 'REUSE'
            elif r.disposition == 'stale':
                status = 'STALE'
            elif not r.success:
                status = 'FAIL'
            elif r.forced:
                status = 'FORCE'
            elif r.invalidation_reason:
                status = 'REDONE'
            else:
                status = 'DONE'
            out = r.output_path or '(no output)'
            err = f'\n    ERROR: {r.error}' if r.error else ''
            reason = f'\n    INVALIDATED: {r.invalidation_reason}' if r.invalidation_reason else ''
            lines.append(f'  [{status}] {r.input_path} -> {out} ({r.duration_ms:.2f} ms){reason}{err}')
            if verbose and r.node_results:
                for nr in r.node_results:
                    nstatus = 'OK' if nr.success else 'FAIL'
                    size = f'{nr.output_size[0]}x{nr.output_size[1]}' if nr.output_size else '?'
                    nerr = f' -> {nr.error}' if nr.error else ''
                    lines.append(f'      + {nstatus} {nr.node_id} ({nr.node_type}, {size}, {nr.duration_ms:.2f} ms){nerr}')
        lines.append('')
    failed_items = [r for r in report.results if not r.success and r.disposition != 'stale']
    if failed_items:
        lines.append('--- Failed Images ---')
        for r in failed_items:
            lines.append(f'  {r.input_path}')
            lines.append(f'    Reason: {r.error}')
        lines.append('')
    stale_items = [r for r in report.results if r.disposition == 'stale']
    if stale_items:
        lines.append('--- Stale Ledger Records (input absent this run) ---')
        for r in stale_items:
            lines.append(f'  {r.input_path}')
        lines.append('')
    return '\n'.join(lines)
