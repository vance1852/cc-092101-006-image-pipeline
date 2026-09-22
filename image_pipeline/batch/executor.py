from typing import Any, Dict, List, Optional, Callable
import os
import json
import time
import traceback
from ..pipeline.engine import PipelineExecutor
from ..utils.types import BatchReport, ImageProcessingResult, ValidationError
from ..utils.image_io import find_images, is_valid_image
from ..nodes.definitions import predict_output_paths
from .ledger import (
    Ledger,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_INVALID,
    STATE_PROCESSING,
    describe_input_file,
    describe_output_file,
)


class BatchExecutor:

    def __init__(self, pipeline_executor: PipelineExecutor, input_dir: str, output_dir: str,
                 config_file: str='', progress_callback: Optional[Callable]=None,
                 resume: bool=True, force: bool=False, retry_failed: bool=False,
                 config_raw: Optional[Dict[str, Any]]=None):
        self.executor = pipeline_executor
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config_file = config_file
        self.progress_callback = progress_callback
        # Resume ledger is always maintained; ``resume`` only controls whether
        # verified prior completations may be reused.
        self.resume = resume
        self.force = force
        self.retry_failed = retry_failed
        self.config_raw = config_raw if config_raw is not None else {}
        self.ledger: Optional[Ledger] = None

    def _ensure_output_dir(self) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def _collect_input_images(self) -> List[str]:
        if not os.path.isdir(self.input_dir):
            raise ValidationError(f"Input directory does not exist: '{self.input_dir}'")
        return find_images(self.input_dir)

    def _planned_outputs(self, img_path: str) -> List[str]:
        filename = os.path.basename(img_path)
        try:
            return predict_output_paths(self.executor.graph, filename, self.output_dir,
                                        self.executor.execution_order)
        except Exception:
            return predict_output_paths(self.executor.graph, filename, self.output_dir)

    def run(self) -> BatchReport:
        report = BatchReport(pipeline_config_file=self.config_file, input_dir=self.input_dir, output_dir=self.output_dir)
        overall_start = time.perf_counter()
        try:
            self._ensure_output_dir()
        except Exception as e:
            report.failed = 0
            report.total = 0
            report.succeeded = 0
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False, error=f'Failed to create output directory: {e}')
            report.results.append(dummy)
            return report

        # ---- Open / recover the ledger BEFORE doing any work --------------
        ledger = Ledger(self.output_dir)
        ledger.open()
        self.ledger = ledger
        run_record = ledger.begin_run(self.config_raw, self.config_file, self.input_dir, self.output_dir)
        report.ledger_path = ledger.path
        report.run_id = run_record['run_id']
        report.ledger_truncated_lines = ledger.truncated_lines
        report.temp_files_removed = ledger.temp_files_removed
        report.recovered_processing = len(ledger.recovered_processing)
        interrupted = set(ledger.interrupted_inputs)
        config_hash = run_record['config_hash']

        try:
            image_paths = self._collect_input_images()
        except ValidationError as e:
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False, error=str(e))
            report.results.append(dummy)
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            return report
        report.total = len(image_paths)
        for idx, img_path in enumerate(image_paths):
            img_result = self._process_one(ledger, img_path, idx, config_hash, interrupted)
            # Aggregate into report.
            if img_result.disposition == 'reused':
                report.reused += 1
                report.skipped += 1
                report.succeeded += 1
            elif img_result.disposition == 'new':
                report.new_completed += 1
                report.succeeded += 1
            elif img_result.disposition == 'invalid':
                report.invalid += 1
                report.failed += 1
            else:
                report.failed += 1
            if img_result.stale:
                report.stale += 1
            report.results.append(img_result)
            if self.progress_callback:
                try:
                    self.progress_callback(idx + 1, report.total, img_result)
                except Exception:
                    pass
        report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
        return report

    # ------------------------------------------------------------------ #
    # Per-item state machine
    # ------------------------------------------------------------------ #

    def _process_one(self, ledger: Ledger, img_path: str, idx: int,
                     config_hash: str, interrupted: set) -> ImageProcessingResult:
        filename = os.path.basename(img_path)
        planned_outputs = self._planned_outputs(img_path)

        # Input identity (byte hash + size + mtime).
        try:
            identity = describe_input_file(img_path)
        except OSError as e:
            error = f'Input file vanished or unreadable before processing: {e}'
            ledger.record_item(img_path, STATE_FAILED, planned_outputs=planned_outputs,
                               error=error)
            return ImageProcessingResult(input_path=img_path, success=False,
                                         error=error, disposition='failed',
                                         planned_outputs=planned_outputs)

        # Pre-check verification: corrupt / unsupported files are INVALID and
        # never reach the pipeline.
        if not is_valid_image(img_path):
            error = 'Image failed pre-check verification (likely corrupt or unsupported format)'
            ledger.record_item(img_path, STATE_INVALID, input_identity=identity,
                               planned_outputs=planned_outputs, error=error)
            return ImageProcessingResult(input_path=img_path, success=False,
                                         error=error, disposition='invalid',
                                         planned_outputs=planned_outputs)

        prior = ledger.latest_item(img_path)
        attempt = int((prior or {}).get('attempt', 0)) + 1

        # --retry-failed means the user explicitly wants a recomputation of
        # items whose latest state is FAILED, so do not let an older valid
        # completion satisfy them.  Items that merely got interrupted
        # (reconciled by crash recovery) may still fall back to an older
        # verified completion -- reusing it is safe and saves the work.
        retry_demanded = (self.retry_failed and prior is not None
                          and prior.get('state') == STATE_FAILED
                          and not prior.get('recovered_processing'))

        # ---- Reuse path: verify bytes + config + every target file -------
        if self.resume and not self.force and not retry_demanded:
            reusable = ledger.reusable_completion(img_path, identity, config_hash, planned_outputs)
            if reusable is not None:
                return self._reuse_item(ledger, img_path, identity, planned_outputs, reusable)

        stale = False
        stale_reason = ''
        if self.resume and not self.force and prior is not None:
            if prior.get('state') == STATE_COMPLETED:
                # A recorded completion no longer verifies -> stale record.
                stale = True
                stale_reason = self._explain_staleness(ledger, img_path, identity,
                                                       config_hash, planned_outputs, prior)
            elif prior.get('state') == STATE_FAILED and not self.retry_failed \
                    and img_path not in interrupted:
                # Carry the failure forward without recomputing.
                error = prior.get('error') or 'Failed in previous run (use --retry-failed to retry)'
                ledger.record_item(img_path, STATE_FAILED, input_identity=identity,
                                   planned_outputs=planned_outputs, error=error,
                                   attempt=attempt, extra={'carried_from_run': prior.get('run_id')})
                result = ImageProcessingResult(input_path=img_path, success=False,
                                               error=error, disposition='failed',
                                               planned_outputs=planned_outputs, attempt=attempt)
                return result

        # ---- Compute path ------------------------------------------------
        ledger.record_item(img_path, STATE_PROCESSING, input_identity=identity,
                           planned_outputs=planned_outputs, attempt=attempt,
                           extra={'stale_reason': stale_reason} if stale else None)
        start = time.perf_counter()
        context: Dict[str, Any] = {'input_path': img_path, 'input_filename': filename,
                                   'output_dir': self.output_dir, 'image_index': idx}
        try:
            img_result = self.executor.run(context)
        except Exception as e:
            img_result = ImageProcessingResult(
                input_path=img_path, success=False,
                error=f'Unexpected error during execution: {e}\n{traceback.format_exc()}')
        img_result.duration_ms = (time.perf_counter() - start) * 1000.0
        img_result.planned_outputs = planned_outputs
        img_result.attempt = attempt

        if not img_result.success:
            ledger.record_item(img_path, STATE_FAILED, input_identity=identity,
                               planned_outputs=planned_outputs,
                               error=img_result.error, duration_ms=img_result.duration_ms,
                               attempt=attempt)
            img_result.disposition = 'failed'
            img_result.stale = stale
            return img_result

        # Success claimed by the pipeline: verify every planned product exists
        # and fingerprint them BEFORE recording COMPLETED.  This is the
        # critical ordering guarantee -- the ledger never claims completion
        # without valid products on disk.
        missing = [p for p in planned_outputs if not os.path.isfile(p)]
        if missing:
            error = f'Pipeline reported success but output file(s) missing: {missing}'
            ledger.record_item(img_path, STATE_FAILED, input_identity=identity,
                               planned_outputs=planned_outputs, error=error,
                               duration_ms=img_result.duration_ms, attempt=attempt)
            img_result.success = False
            img_result.error = error
            img_result.disposition = 'failed'
            img_result.stale = stale
            return img_result
        try:
            output_identity = [describe_output_file(p) for p in planned_outputs]
        except OSError as e:
            error = f'Failed to fingerprint output after processing: {e}'
            ledger.record_item(img_path, STATE_FAILED, input_identity=identity,
                               planned_outputs=planned_outputs, error=error,
                               duration_ms=img_result.duration_ms, attempt=attempt)
            img_result.success = False
            img_result.error = error
            img_result.disposition = 'failed'
            img_result.stale = stale
            return img_result

        ledger.record_item(img_path, STATE_COMPLETED, input_identity=identity,
                           planned_outputs=planned_outputs, output_identity=output_identity,
                           duration_ms=img_result.duration_ms, attempt=attempt)
        img_result.disposition = 'new'
        img_result.stale = stale
        if stale:
            img_result.error = stale_reason
        if img_result.output_path is None and planned_outputs:
            img_result.output_path = planned_outputs[0]
        return img_result

    def _reuse_item(self, ledger: Ledger, img_path: str, identity: Dict[str, Any],
                    planned_outputs: List[str], reusable: Dict[str, Any]) -> ImageProcessingResult:
        # Re-fingerprint the on-disk products and persist the verified reuse
        # as a COMPLETED line in the current run.
        output_identity = [describe_output_file(p) for p in planned_outputs]
        ledger.record_item(img_path, STATE_COMPLETED, input_identity=identity,
                           planned_outputs=planned_outputs, output_identity=output_identity,
                           extra={'reused_from_run': reusable.get('run_id')})
        return ImageProcessingResult(
            input_path=img_path,
            output_path=planned_outputs[0] if planned_outputs else reusable.get('planned_outputs', [None])[0],
            success=True, disposition='reused', reused=True,
            planned_outputs=planned_outputs)

    def _explain_staleness(self, ledger: Ledger, img_path: str, identity: Dict[str, Any],
                           config_hash: str, planned_outputs: List[str],
                           prior: Dict[str, Any]) -> str:
        run = ledger.runs.get(prior.get('run_id'), {})
        reasons = []
        if run.get('config_hash') != config_hash:
            reasons.append('pipeline config changed')
        rec_input = prior.get('input') or {}
        if rec_input.get('sha256') != identity.get('sha256'):
            reasons.append('input file bytes changed')
        rec_outputs = {o.get('path'): o for o in (prior.get('outputs') or [])}
        for p in planned_outputs:
            orec = rec_outputs.get(p)
            if orec is None:
                reasons.append(f'planned output changed: {os.path.basename(p)}')
            elif not os.path.isfile(p):
                reasons.append(f'output missing: {os.path.basename(p)}')
            else:
                try:
                    actual = describe_output_file(p)
                    if actual['sha256'] != orec.get('sha256'):
                        reasons.append(f'output bytes changed: {os.path.basename(p)}')
                except OSError:
                    reasons.append(f'output unreadable: {os.path.basename(p)}')
        if not reasons:
            reasons.append('recorded completion failed verification')
        return 'stale record recomputed (' + '; '.join(reasons) + ')'

    def write_report(self, report: BatchReport, path: str=None) -> str:
        if path is None:
            path = os.path.join(self.output_dir, 'batch_report.json')
        report_dir = os.path.dirname(path)
        if report_dir and (not os.path.exists(report_dir)):
            os.makedirs(report_dir, exist_ok=True)
        from ..utils.image_io import write_file_atomic
        payload = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        write_file_atomic(path, lambda tmp: _write_text(tmp, payload))
        return path


def _write_text(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def print_text_report(report: BatchReport, verbose: bool=False) -> str:
    lines = []
    lines.append('=' * 60)
    lines.append('BATCH PROCESSING REPORT')
    lines.append('=' * 60)
    lines.append(f'Pipeline config : {report.pipeline_config_file}')
    lines.append(f'Input directory : {report.input_dir}')
    lines.append(f'Output directory: {report.output_dir}')
    if report.run_id:
        lines.append(f'Run ID          : {report.run_id}')
        lines.append(f'Ledger          : {report.ledger_path}')
    lines.append('')
    lines.append('--- Summary ---')
    lines.append(f'Total images    : {report.total}')
    lines.append(f'Succeeded       : {report.succeeded}')
    lines.append(f'  newly done    : {report.new_completed}')
    lines.append(f'  reused        : {report.reused}')
    lines.append(f'Failed          : {report.failed}')
    lines.append(f'Invalid inputs  : {report.invalid}')
    lines.append(f'Stale records   : {report.stale}')
    lines.append(f'Skipped         : {report.skipped}')
    lines.append(f'Total duration  : {report.total_duration_ms:.2f} ms')
    if report.total > 0:
        lines.append(f'Avg per image   : {report.total_duration_ms / report.total:.2f} ms')
    recovered = (report.ledger_truncated_lines, report.recovered_processing, report.temp_files_removed)
    if any(recovered):
        lines.append('')
        lines.append('--- Crash Recovery ---')
        lines.append(f'Torn ledger lines truncated : {report.ledger_truncated_lines}')
        lines.append(f'Interrupted items retried   : {report.recovered_processing}')
        lines.append(f'Leftover temp files removed : {report.temp_files_removed}')
    lines.append('')
    if verbose:
        lines.append('--- Per-Image Details ---')
        for r in report.results:
            tag = r.disposition.upper() if r.disposition else ('OK' if r.success else 'FAIL')
            out = r.output_path or '(no output)'
            err = f'\n    ERROR: {r.error}' if r.error else ''
            lines.append(f'  [{tag:6s}] {r.input_path} -> {out} ({r.duration_ms:.2f} ms){err}')
            if verbose and r.node_results:
                for nr in r.node_results:
                    nstatus = 'OK' if nr.success else 'FAIL'
                    size = f'{nr.output_size[0]}x{nr.output_size[1]}' if nr.output_size else '?'
                    nerr = f' -> {nr.error}' if nr.error else ''
                    lines.append(f'      + {nstatus} {nr.node_id} ({nr.node_type}, {size}, {nr.duration_ms:.2f} ms){nerr}')
        lines.append('')
    failed = [r for r in report.results if not r.success]
    if failed:
        lines.append('--- Failed / Invalid Images ---')
        for r in failed:
            lines.append(f'  [{r.disposition or "FAIL"}] {r.input_path}')
            lines.append(f'    Reason: {r.error}')
        lines.append('')
    return '\n'.join(lines)
