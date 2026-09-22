"""Recoverable run ledger for batch processing.

The ledger is an append-only JSONL file (one JSON object per line) living in
``<output_dir>/.imgpipe/ledger.jsonl``.  It records, for every run:

* ``run``       -- run id, config fingerprint (hash + full content), input and
                   output directories, timestamp;
* ``item``      -- one per input image: its identity (size / mtime_ns /
                   SHA-256 of the bytes), the planned output files and the
                   state transitions PROCESSING -> COMPLETED / FAILED /
                   INVALID (one PROCESSING line, then one terminal line).

Durability rules
----------------
* Output images are written atomically (temp file + ``os.replace``) *before*
  the COMPLETED line is appended.  A COMPLETED record therefore never exists
  without a valid product on disk.
* Every append is fsynced before the corresponding in-memory state is
  considered committed.
* A torn trailing line (power loss mid-append) is detected on open, truncated
  away and reported.
* Items left in PROCESSING by a dead process, and leftover temp files, are
  reconciled back to a retryable state at startup.

An item from a previous run is reused (skipped) only when the input bytes,
the relevant config and every planned target file all match the recorded
COMPLETED entry.
"""

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..utils.image_io import (
    canonical_json_sha256,
    cleanup_temp_files,
    file_sha256,
    write_file_atomic,
)

#: Item lifecycle states recorded in the ledger.
STATE_PENDING = 'pending'
STATE_PROCESSING = 'processing'
STATE_COMPLETED = 'completed'
STATE_FAILED = 'failed'
STATE_INVALID = 'invalid'

LEDGER_DIRNAME = '.imgpipe'
LEDGER_FILENAME = 'ledger.jsonl'
LEDGER_VERSION = 1


class LedgerError(Exception):
    pass


# ---------------------------------------------------------------------------
# Fingerprinting helpers
# ---------------------------------------------------------------------------

def config_fingerprint(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a stable fingerprint of the *effective* pipeline configuration.

    Only content that can affect pixels or output file names is included
    (nodes with their effective params, and edges); cosmetic fields such as
    ``name`` / ``description`` are intentionally ignored.
    """
    nodes = raw_config.get('nodes', []) or []
    node_specs = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        node_specs.append({
            'id': n.get('id'),
            'type': n.get('type'),
            'params': n.get('params', {}) or {},
        })
    node_specs.sort(key=lambda s: (str(s.get('id')), str(s.get('type'))))
    edges = raw_config.get('edges', []) or []
    edge_specs = sorted(
        [(e.get('from'), e.get('to'), e.get('slot', 0)) for e in edges if isinstance(e, dict)],
        key=lambda e: (str(e[0]), str(e[1]), int(e[2])),
    )
    relevant = {
        'ledger_version': LEDGER_VERSION,
        'nodes': node_specs,
        'edges': edge_specs,
        'defaults': raw_config.get('defaults', {}) or {},
    }
    digest = canonical_json_sha256(relevant)
    return {
        'hash': digest,
        'content': raw_config,
        'relevant': relevant,
    }


def describe_input_file(path: str) -> Dict[str, Any]:
    """Identity record for an input file: byte hash plus size/mtime metadata."""
    st = os.stat(path)
    return {
        'path': path,
        'size': st.st_size,
        'mtime_ns': st.st_mtime_ns,
        'sha256': file_sha256(path),
    }


def describe_output_file(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    return {
        'path': path,
        'size': st.st_size,
        'sha256': file_sha256(path),
    }


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class Ledger:
    """Append-only JSONL ledger with crash recovery."""

    def __init__(self, output_dir: str):
        self.output_dir = os.path.abspath(output_dir)
        self.dir = os.path.join(self.output_dir, LEDGER_DIRNAME)
        self.path = os.path.join(self.dir, LEDGER_FILENAME)
        self.run_id: str = ''
        # run_id -> {config_hash, input_dir, output_dir}
        self.runs: Dict[str, Dict[str, Any]] = {}
        # (run_id, input_path) -> latest item record
        self.items: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # input_path -> list of run_ids that ever recorded it (oldest first)
        self._history: Dict[str, List[str]] = {}
        self.recovered_processing: List[Dict[str, Any]] = []
        self.truncated_lines: int = 0
        self.temp_files_removed: int = 0

    # -- open / recovery ---------------------------------------------------

    def open(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        self.temp_files_removed = cleanup_temp_files(self.output_dir)
        if os.path.exists(self.path):
            self._load_and_repair()
        self._reconcile_processing()

    def _load_and_repair(self) -> None:
        """Read every complete JSON line; truncate a torn tail in place."""
        with open(self.path, 'rb') as f:
            raw = f.read()
        if not raw:
            return
        # Split keeping track of byte offsets so a torn tail can be cut off.
        good_bytes = bytearray()
        truncated = 0
        for line_bytes in raw.splitlines(keepends=True):
            stripped = line_bytes.strip()
            if not stripped:
                good_bytes.extend(line_bytes)
                continue
            try:
                record = json.loads(stripped.decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                # Torn / partial write at tail: drop this line and everything
                # after it (it must be the last line of an append-only file).
                truncated += 1
                break
            if not isinstance(record, dict) or 'type' not in record:
                truncated += 1
                break
            self._ingest(record)
            good_bytes.extend(line_bytes)
        if truncated and len(good_bytes) != len(raw):
            def _rewrite(tmp_path: str) -> None:
                with open(tmp_path, 'wb') as f:
                    f.write(bytes(good_bytes))
            write_file_atomic(self.path, _rewrite)
        self.truncated_lines = truncated

    def _ingest(self, record: Dict[str, Any]) -> None:
        rtype = record.get('type')
        if rtype == 'run':
            rid = record.get('run_id')
            if rid:
                self.runs[rid] = record
        elif rtype == 'item':
            rid = record.get('run_id')
            ipath = record.get('input_path')
            if rid and ipath is not None:
                key = (rid, ipath)
                # Keep the newest record per (run, item): later lines are
                # state transitions for the same item.
                self.items[key] = record
                hist = self._history.setdefault(ipath, [])
                if rid not in hist:
                    hist.append(rid)

    def _reconcile_processing(self) -> None:
        """Any item whose newest state is PROCESSING died mid-run.

        Append a reconciled FAILED line so the ledger itself reflects the
        convergence, and remember them for the current run's report.
        """
        for (rid, ipath), rec in list(self.items.items()):
            if rec.get('state') == STATE_PROCESSING:
                note = 'recovered: interrupted while processing (crash/restart)'
                reconciled = dict(rec)
                reconciled['state'] = STATE_FAILED
                reconciled['error'] = note
                reconciled['recovered_processing'] = True
                reconciled['ts'] = time.time()
                self._append_record(reconciled)
                self.items[(rid, ipath)] = reconciled
                self.recovered_processing.append(reconciled)

    # -- run lifecycle -----------------------------------------------------

    def begin_run(
        self,
        config: Dict[str, Any],
        config_path: str,
        input_dir: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        fp = config_fingerprint(config)
        self.run_id = f'{time.strftime("%Y%m%dT%H%M%S")}-{uuid.uuid4().hex[:12]}'
        record = {
            'type': 'run',
            'run_id': self.run_id,
            'ts': time.time(),
            'config_path': os.path.abspath(config_path) if config_path else '',
            'config_hash': fp['hash'],
            'config': fp['content'],
            'config_relevant': fp['relevant'],
            'input_dir': os.path.abspath(input_dir),
            'output_dir': os.path.abspath(output_dir),
        }
        self._append_record(record)
        self.runs[self.run_id] = record
        return record

    # -- item transitions --------------------------------------------------

    def record_item(
        self,
        input_path: str,
        state: str,
        *,
        input_identity: Optional[Dict[str, Any]]=None,
        planned_outputs: Optional[List[str]]=None,
        output_identity: Optional[List[Dict[str, Any]]]=None,
        error: Optional[str]=None,
        duration_ms: Optional[float]=None,
        attempt: int=1,
        extra: Optional[Dict[str, Any]]=None,
    ) -> Dict[str, Any]:
        if not self.run_id:
            raise LedgerError('begin_run() must be called before recording items')
        record: Dict[str, Any] = {
            'type': 'item',
            'run_id': self.run_id,
            'ts': time.time(),
            'input_path': input_path,
            'state': state,
            'attempt': attempt,
        }
        if input_identity is not None:
            record['input'] = input_identity
        if planned_outputs is not None:
            record['planned_outputs'] = planned_outputs
        if output_identity is not None:
            record['outputs'] = output_identity
        if error is not None:
            record['error'] = error
        if duration_ms is not None:
            record['duration_ms'] = round(float(duration_ms), 2)
        if extra:
            record.update(extra)
        self._append_record(record)
        self.items[(self.run_id, input_path)] = record
        self._history.setdefault(input_path, []).append(self.run_id)
        return record

    # -- reuse lookup ------------------------------------------------------

    def reusable_completion(
        self,
        input_path: str,
        input_identity: Dict[str, Any],
        config_hash: str,
        planned_outputs: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Return a prior COMPLETED record that can safely be reused.

        Skip is allowed only when ALL of the following hold:

        1. the input bytes hash (and size) match the recorded identity;
        2. the previous run used a config with the same fingerprint;
        3. every planned target file exists and matches its recorded
           size + byte hash.
        """
        for rid in reversed(self._history.get(input_path, ())):
            rec = self.items.get((rid, input_path))
            if not rec or rec.get('state') != STATE_COMPLETED:
                continue
            run = self.runs.get(rid)
            if not run or run.get('config_hash') != config_hash:
                continue
            rec_input = rec.get('input') or {}
            if rec_input.get('sha256') != input_identity.get('sha256'):
                continue
            if rec_input.get('size') != input_identity.get('size'):
                continue
            rec_outputs = rec.get('outputs') or []
            rec_by_path = {o.get('path'): o for o in rec_outputs}
            if not rec_outputs or len(rec_outputs) != len(planned_outputs):
                continue
            all_match = True
            for out_path in planned_outputs:
                orec = rec_by_path.get(out_path)
                if orec is None:
                    all_match = False
                    break
                if not os.path.isfile(out_path):
                    all_match = False
                    break
                try:
                    actual = describe_output_file(out_path)
                except OSError:
                    all_match = False
                    break
                if actual['size'] != orec.get('size') or actual['sha256'] != orec.get('sha256'):
                    all_match = False
                    break
            if all_match:
                return rec
        return None

    def prior_failure(self, input_path: str, config_hash: str) -> Optional[Dict[str, Any]]:
        """Most recent record for this input under the same config, if failed."""
        for rid in reversed(self._history.get(input_path, ())):
            run = self.runs.get(rid)
            if not run or run.get('config_hash') != config_hash:
                continue
            rec = self.items.get((rid, input_path))
            if rec and rec.get('state') == STATE_FAILED:
                return rec
        return None

    def latest_item(self, input_path: str) -> Optional[Dict[str, Any]]:
        """Newest ledger item ever recorded for ``input_path`` (any config)."""
        for rid in reversed(self._history.get(input_path, ())):
            rec = self.items.get((rid, input_path))
            if rec is not None:
                return rec
        return None

    @property
    def interrupted_inputs(self) -> List[str]:
        """Inputs whose last run was interrupted mid-processing (any config)."""
        return [rec.get('input_path') for rec in self.recovered_processing]

    def all_completed_items(self) -> List[Dict[str, Any]]:
        """Newest COMPLETED record per input path (for stale-run inspection)."""
        result: List[Dict[str, Any]] = []
        for ipath, rids in self._history.items():
            for rid in reversed(rids):
                rec = self.items.get((rid, ipath))
                if rec and rec.get('state') == STATE_COMPLETED:
                    result.append(rec)
                    break
        return result

    # -- low level append --------------------------------------------------

    def _append_record(self, record: Dict[str, Any]) -> None:
        os.makedirs(self.dir, exist_ok=True)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        data = (line + '\n').encode('utf-8')
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            dfd = os.open(self.dir, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
