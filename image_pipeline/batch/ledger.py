"""Resumable run ledger for batch processing.

The ledger is an append-only, newline-delimited JSON event log. Every run
persists the full pipeline configuration content, every input file's byte
identity (size + sha256), the planned output targets, and each image's state
transitions (plan -> processing -> completed/failed).

A COMPLETED record is written *after* the output artifact has been committed
atomically (temp file -> fsync -> os.replace -> directory fsync) and re-hashed,
so there is never a window in which the ledger claims success without a valid
product on disk:

* a crash before the completed event      -> last state is "processing" -> retry
* a crash while appending the event       -> corrupt tail is truncated -> retry
* a crash during the write                -> stale *.imgpipe.tmp is removed -> retry

A previous result is reused (skipped) only when input bytes, the relevant
configuration hash, and every recorded target file (size + sha256) all match
the current plan. Any mismatch invalidates the old record and forces a retry.
"""
from typing import Any, Dict, List, Optional
import glob
import hashlib
import json
import os
import time
import uuid

LEDGER_VERSION = '1'
LEDGER_DIR_NAME = '.imgpipe'
LEDGER_FILE_NAME = 'ledger.jsonl'
TEMP_SUFFIX = 'imgpipe.tmp'

STATE_PROCESSING = 'processing'
STATE_COMPLETED = 'completed'
STATE_FAILED = 'failed'

# Reasons an existing ledger record is no longer trustworthy.
REASON_CONFIG_CHANGED = 'config_changed'
REASON_INPUT_CHANGED = 'input_changed'
REASON_OUTPUT_PLAN_CHANGED = 'output_plan_changed'
REASON_OUTPUT_MISSING = 'output_missing'
REASON_OUTPUT_CHANGED = 'output_changed'
REASON_INPUT_VANISHED = 'input_vanished'
REASON_INTERRUPTED = 'interrupted'

_CHUNK = 1024 * 1024


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_config_json(config_raw: Dict[str, Any]) -> str:
    """Stable serialization of configuration content.

    Two configs that carry exactly the same nodes/edges/params produce the same
    string regardless of key order or insignificant whitespace.
    """
    return json.dumps(config_raw, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def config_hash(config_raw: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_config_json(config_raw).encode('utf-8')).hexdigest()


def default_ledger_path(output_dir: str) -> str:
    return os.path.join(output_dir, LEDGER_DIR_NAME, LEDGER_FILE_NAME)


def fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: str) -> None:
    """Best-effort persistence of a directory entry change (e.g. rename)."""
    if not path:
        path = '.'
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems do not support fsync on directories.
        pass
    finally:
        os.close(fd)


def is_temp_file_name(name: str) -> bool:
    return name.endswith('.' + TEMP_SUFFIX)


def cleanup_temp_files(directories: List[str]) -> List[str]:
    """Remove interrupted atomic-write temp files left by earlier runs."""
    removed: List[str] = []
    for directory in dict.fromkeys(directories):
        if not directory or not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not is_temp_file_name(name):
                continue
            path = os.path.join(directory, name)
            try:
                os.remove(path)
                removed.append(path)
            except OSError:
                pass
    return removed


class AtomicImageWriter:
    """Write images through a temp file + atomic rename.

    Drop-in callable with the same shape as ``utils.image_io.write_image``.
    The final path never observes a partial file.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.renamed: List[str] = []

    def temp_path_for(self, out_path: str) -> str:
        directory = os.path.dirname(out_path) or '.'
        base = os.path.basename(out_path)
        return os.path.join(directory, f'.{base}.{self.run_id}.{TEMP_SUFFIX}')

    def __call__(self, img: Any, out_path: str, fmt: Optional[str] = None, quality: int = 90) -> None:
        from ..utils.image_io import write_image
        directory = os.path.dirname(out_path) or '.'
        os.makedirs(directory, exist_ok=True)
        tmp_path = self.temp_path_for(out_path)
        try:
            write_image(img, tmp_path, fmt=fmt, quality=quality)
            fsync_file(tmp_path)
            os.replace(tmp_path, out_path)
            fsync_dir(directory)
            self.renamed.append(out_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise


class _ItemRecord:
    """Replayed view of one input item's latest ledger state."""

    __slots__ = ('key', 'plan', 'last_state', 'last_error', 'completed',
                 'last_run_id', 'failed')

    def __init__(self, key: str):
        self.key = key
        self.plan: Optional[Dict[str, Any]] = None
        self.last_state: Optional[str] = None
        self.last_error: Optional[str] = None
        self.completed: Optional[Dict[str, Any]] = None
        self.failed: Optional[Dict[str, Any]] = None
        self.last_run_id: Optional[str] = None


class Ledger:
    """Append-only resumable ledger. Use :meth:`open` to construct."""

    def __init__(self, path: str, run_id: str):
        self.path = path
        self.run_id = run_id
        self._file: Optional[Any] = None
        self.records: Dict[str, _ItemRecord] = {}
        # Recovery statistics gathered while opening the log.
        self.recovered_interrupted = 0
        self.truncated_corrupt_tail = False
        self.tail_dropped_bytes = 0
        self.temp_files_removed: List[str] = []

    # ------------------------------------------------------------------ open

    @classmethod
    def open(cls, output_dir: str, path: Optional[str] = None) -> 'Ledger':
        ledger_path = path or default_ledger_path(output_dir)
        run_id = f'{os.getpid()}-{uuid.uuid4().hex[:12]}'
        ledger = cls(ledger_path, run_id)
        os.makedirs(os.path.dirname(ledger_path) or '.', exist_ok=True)
        # Sweep interrupted atomic writes (images and repaired ledger) left by
        # earlier crashes before anything new starts.
        ledger.temp_files_removed = cleanup_temp_files(
            [output_dir, os.path.dirname(ledger_path)])
        events = ledger._read_and_recover()
        ledger._replay(events)
        ledger._open_for_append()
        return ledger

    def _read_and_recover(self) -> List[Dict[str, Any]]:
        """Read all events, truncating (and physically removing) a torn tail.

        A partial write or power loss can leave a malformed final line. Every
        complete line before it is still valid; the garbage suffix is pruned so
        future appends attach to a sound boundary.
        """
        if not os.path.isfile(self.path):
            return []
        with open(self.path, 'rb') as f:
            data = f.read()
        events: List[Dict[str, Any]] = []
        good = bytearray()
        cursor = 0
        corrupted = False
        needs_rewrite = False
        while cursor < len(data):
            nl = data.find(b'\n', cursor)
            had_newline = nl != -1
            if nl == -1:
                line = data[cursor:]
                next_cursor = len(data)
            else:
                line = data[cursor:nl]
                next_cursor = nl + 1
            stripped = line.strip()
            event: Optional[Dict[str, Any]] = None
            if stripped:
                try:
                    parsed = json.loads(stripped.decode('utf-8'))
                    if isinstance(parsed, dict) and parsed.get('t'):
                        event = parsed
                    else:
                        corrupted = True
                except (ValueError, UnicodeDecodeError):
                    corrupted = True
            if corrupted:
                # Everything from the start of this line to EOF is a torn tail.
                break
            if event is not None:
                events.append(event)
            good.extend(line)
            good.extend(b'\n')
            if not had_newline:
                # Valid unterminated final line: re-terminate so future appends
                # start on a fresh line.
                needs_rewrite = True
            cursor = next_cursor
        if corrupted or needs_rewrite or len(good) != len(data):
            self.truncated_corrupt_tail = corrupted
            self.tail_dropped_bytes = len(data) - cursor
            self._rewrite_log(bytes(good))
        return events

    def _rewrite_log(self, content: bytes) -> None:
        directory = os.path.dirname(self.path) or '.'
        os.makedirs(directory, exist_ok=True)
        tmp_path = os.path.join(directory, f'.{os.path.basename(self.path)}.repair.{TEMP_SUFFIX}')
        with open(tmp_path, 'wb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)
        fsync_dir(directory)

    def _replay(self, events: List[Dict[str, Any]]) -> None:
        for ev in events:
            kind = ev.get('t')
            if kind == 'plan':
                key = ev.get('item')
                if not key:
                    continue
                rec = self.records.setdefault(key, _ItemRecord(key))
                rec.plan = ev
                rec.last_run_id = ev.get('run_id')
            elif kind == 'state':
                key = ev.get('item')
                if not key:
                    continue
                rec = self.records.setdefault(key, _ItemRecord(key))
                rec.last_run_id = ev.get('run_id')
                state = ev.get('state')
                rec.last_state = state
                if state == STATE_COMPLETED:
                    rec.completed = ev
                    rec.failed = None
                    rec.last_error = None
                elif state == STATE_FAILED:
                    rec.failed = ev
                    rec.completed = None
                    rec.last_error = ev.get('error')
        self.recovered_interrupted = sum(
            1 for rec in self.records.values() if rec.last_state == STATE_PROCESSING
        )

    def _open_for_append(self) -> None:
        directory = os.path.dirname(self.path) or '.'
        os.makedirs(directory, exist_ok=True)
        self._file = open(self.path, 'a', encoding='utf-8', newline='\n')

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            except (OSError, ValueError):
                pass
            self._file.close()
            self._file = None

    def __enter__(self) -> 'Ledger':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------------------------------------------------------------- events

    def _emit(self, event: Dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError('Ledger is closed')
        line = json.dumps(event, ensure_ascii=False, separators=(',', ':'))
        self._file.write(line + '\n')
        self._file.flush()
        os.fsync(self._file.fileno())

    def run_started(self, config_raw: Dict[str, Any], cfg_hash: str, input_dir: str,
                    output_dir: str, config_file: str, app_version: str,
                    mode: str) -> None:
        self._emit({
            't': 'run_start',
            'ver': LEDGER_VERSION,
            'run_id': self.run_id,
            'ts': time.time(),
            'app_version': app_version,
            'mode': mode,
            'config_hash': cfg_hash,
            'config_file': config_file,
            'config': config_raw,
            'input_dir': os.path.abspath(input_dir),
            'output_dir': os.path.abspath(output_dir),
        })

    def item_planned(self, item_key: str, input_identity: Dict[str, Any],
                     outputs: List[Dict[str, str]]) -> None:
        self._emit({
            't': 'plan',
            'ver': LEDGER_VERSION,
            'run_id': self.run_id,
            'ts': time.time(),
            'item': item_key,
            'input': input_identity,
            'outputs': outputs,
        })

    def mark_processing(self, item_key: str) -> None:
        self._emit({'t': 'state', 'run_id': self.run_id, 'ts': time.time(),
                    'item': item_key, 'state': STATE_PROCESSING})

    def mark_completed(self, item_key: str, cfg_hash: str,
                       outputs: List[Dict[str, Any]],
                       input_sha: Optional[str] = None) -> None:
        event = {'t': 'state', 'run_id': self.run_id, 'ts': time.time(),
                 'item': item_key, 'state': STATE_COMPLETED,
                 'config_hash': cfg_hash, 'outputs': outputs}
        if input_sha is not None:
            event['input_sha256'] = input_sha
        self._emit(event)

    def mark_failed(self, item_key: str, error: str,
                    cfg_hash: Optional[str] = None,
                    input_sha: Optional[str] = None) -> None:
        event = {'t': 'state', 'run_id': self.run_id, 'ts': time.time(),
                 'item': item_key, 'state': STATE_FAILED, 'error': error}
        if cfg_hash is not None:
            event['config_hash'] = cfg_hash
        if input_sha is not None:
            event['input_sha256'] = input_sha
        self._emit(event)

    def run_finished(self) -> None:
        self._emit({'t': 'run_end', 'run_id': self.run_id, 'ts': time.time()})

    # --------------------------------------------------------- decisions

    def prior_record(self, item_key: str) -> Optional[_ItemRecord]:
        return self.records.get(item_key)

    def reuse_invalidation(self, item_key: str, input_sha: str,
                           planned_outputs: List[Dict[str, str]],
                           cfg_hash: str, output_dir: str) -> Optional[str]:
        """Return None when the item may be reused, else an invalidation reason.

        The completed event is self-contained (config hash, input hash and
        output identities), so reuse never relies on separate plan records.
        """
        rec = self.records.get(item_key)
        if rec is None or rec.last_state != STATE_COMPLETED or rec.completed is None:
            return None
        ev = rec.completed
        if ev.get('config_hash') != cfg_hash:
            return REASON_CONFIG_CHANGED
        if ev.get('input_sha256') != input_sha:
            return REASON_INPUT_CHANGED
        identities = ev.get('outputs', [])
        prior_targets = [o.get('path') for o in identities]
        current_targets = [o['path'] for o in planned_outputs]
        if prior_targets != current_targets:
            return REASON_OUTPUT_PLAN_CHANGED
        for ident in identities:
            rel = ident.get('path')
            abs_path = os.path.join(output_dir, rel)
            if not os.path.isfile(abs_path):
                return REASON_OUTPUT_MISSING
            try:
                size = os.path.getsize(abs_path)
                if size == 0 or size != ident.get('size'):
                    return REASON_OUTPUT_CHANGED
                if sha256_file(abs_path) != ident.get('sha256'):
                    return REASON_OUTPUT_CHANGED
            except OSError:
                return REASON_OUTPUT_MISSING
        return None

    def was_failed(self, item_key: str) -> bool:
        rec = self.records.get(item_key)
        return rec is not None and rec.last_state == STATE_FAILED

    def failure_is_current(self, item_key: str, input_sha: str, cfg_hash: str) -> bool:
        """Whether the last failure was recorded against these exact bytes/config."""
        rec = self.records.get(item_key)
        if rec is None or rec.failed is None:
            return False
        ev = rec.failed
        return ev.get('config_hash') == cfg_hash and ev.get('input_sha256') == input_sha

    def failure_staleness_reason(self, item_key: str, input_sha: str,
                                 cfg_hash: str) -> Optional[str]:
        rec = self.records.get(item_key)
        if rec is None or rec.failed is None:
            return None
        ev = rec.failed
        if ev.get('config_hash') != cfg_hash:
            return REASON_CONFIG_CHANGED
        if ev.get('input_sha256') != input_sha:
            return REASON_INPUT_CHANGED
        return None

    def failure_error(self, item_key: str) -> Optional[str]:
        rec = self.records.get(item_key)
        return rec.last_error if rec else None

    def completed_outputs(self, item_key: str) -> List[Dict[str, Any]]:
        rec = self.records.get(item_key)
        if rec is None or rec.completed is None:
            return []
        return rec.completed.get('outputs', [])

    def orphan_keys(self, current_keys: set) -> List[str]:
        """Known ledger items whose input is absent from the current run.

        Based on the cumulative item set, so a vanished input is reported even
        if the previous run crashed before planning anything.
        """
        return sorted(set(self.records.keys()) - current_keys)
