from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

class NodeType(str, Enum):
    INPUT = 'input'
    GRAYSCALE = 'grayscale'
    BRIGHTNESS = 'brightness'
    CONTRAST = 'contrast'
    THRESHOLD = 'threshold'
    BOX_BLUR = 'box_blur'
    GAUSSIAN_BLUR = 'gaussian_blur'
    SHARPEN = 'sharpen'
    SOBEL = 'sobel'
    PREWITT = 'prewitt'
    CROP = 'crop'
    RESIZE = 'resize'
    OUTPUT = 'output'

class PipelineError(Exception):
    pass

class ValidationError(PipelineError):
    pass

class ExecutionError(PipelineError):
    pass

@dataclass
class ValidationIssue:
    level: str
    message: str
    node_id: Optional[str] = None
    field: Optional[str] = None

    def __str__(self):
        parts = [f'[{self.level.upper()}]']
        if self.node_id:
            parts.append(f'(node={self.node_id})')
        if self.field:
            parts.append(f'[field={self.field}]')
        parts.append(self.message)
        return ' '.join(parts)

@dataclass
class ValidationResult:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all((i.level != 'error' for i in self.issues))

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == 'error']

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == 'warning']

    def add_error(self, msg: str, node_id: str=None, field: str=None):
        self.issues.append(ValidationIssue('error', msg, node_id, field))

    def add_warning(self, msg: str, node_id: str=None, field: str=None):
        self.issues.append(ValidationIssue('warning', msg, node_id, field))

    def extend(self, other: 'ValidationResult'):
        self.issues.extend(other.issues)

    def __str__(self):
        lines = [f"Validation: {('PASS' if self.valid else 'FAIL')}"]
        for i in self.issues:
            lines.append(f'  {str(i)}')
        return '\n'.join(lines)

@dataclass
class NodeExecutionResult:
    node_id: str
    node_type: str
    success: bool
    duration_ms: float = 0.0
    error: Optional[str] = None
    output_size: Optional[tuple] = None

@dataclass
class ImageProcessingResult:
    input_path: str
    output_path: Optional[str] = None
    success: bool = False
    duration_ms: float = 0.0
    error: Optional[str] = None
    node_results: List[NodeExecutionResult] = field(default_factory=list)
    #: Ledger disposition: 'new' | 'reused' | 'failed' | 'invalid' | ''
    disposition: str = ''
    #: True when the result was reused from a previous run (no recompute).
    reused: bool = False
    #: Planned output paths for this input (all output nodes).
    planned_outputs: List[str] = field(default_factory=list)
    #: Attempt number within the current run (retries).
    attempt: int = 1
    #: A previously completed record was found stale and got recomputed.
    stale: bool = False

@dataclass
class BatchReport:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration_ms: float = 0.0
    results: List[ImageProcessingResult] = field(default_factory=list)
    pipeline_config_file: str = ''
    input_dir: str = ''
    output_dir: str = ''
    # --- resumable-ledger fields ---
    ledger_path: str = ''
    run_id: str = ''
    #: Freshly processed and committed in this run.
    new_completed: int = 0
    #: Reused from a prior run after full identity verification.
    reused: int = 0
    #: Inputs that failed pre-check verification (corrupt/unsupported).
    invalid: int = 0
    #: Items whose recorded completion no longer matches disk/config/input.
    stale: int = 0
    #: Items reconciled from a dangling PROCESSING state at startup.
    recovered_processing: int = 0
    #: Torn ledger lines truncated at startup.
    ledger_truncated_lines: int = 0
    #: Leftover temp files swept at startup.
    temp_files_removed: int = 0

    @property
    def new_or_reused(self) -> int:
        return self.new_completed + self.reused

    def to_dict(self) -> Dict[str, Any]:
        return {'summary': {'total': self.total, 'succeeded': self.succeeded, 'failed': self.failed, 'skipped': self.skipped, 'new_completed': self.new_completed, 'reused': self.reused, 'invalid': self.invalid, 'stale': self.stale, 'total_duration_ms': round(self.total_duration_ms, 2)}, 'ledger': {'run_id': self.run_id, 'path': self.ledger_path, 'recovered_processing': self.recovered_processing, 'truncated_lines': self.ledger_truncated_lines, 'temp_files_removed': self.temp_files_removed}, 'config': {'pipeline_file': self.pipeline_config_file, 'input_dir': self.input_dir, 'output_dir': self.output_dir}, 'results': [{'input': r.input_path, 'output': r.output_path, 'success': r.success, 'reused': r.reused, 'disposition': r.disposition, 'attempt': r.attempt, 'planned_outputs': r.planned_outputs, 'duration_ms': round(r.duration_ms, 2), 'error': r.error, 'nodes': [{'node_id': nr.node_id, 'node_type': nr.node_type, 'success': nr.success, 'duration_ms': round(nr.duration_ms, 2), 'error': nr.error, 'output_size': list(nr.output_size) if nr.output_size else None} for nr in r.node_results]} for r in self.results]}
