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
    output_paths: List[str] = field(default_factory=list)
    # Run disposition: 'completed' (newly done this run), 'reused' (skipped
    # because the ledger and artifacts matched), 'failed', 'invalidated'
    # (a stale ledger record was detected and reprocessed), 'stale' (ledger
    # record references an input no longer present this run).
    disposition: Optional[str] = None
    invalidation_reason: Optional[str] = None
    forced: bool = False
    reused_from_run: Optional[str] = None
    attempted: bool = True

    @property
    def reused(self) -> bool:
        return self.disposition == 'reused'

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
    # Resumable-run receipt counts.
    newly_completed: int = 0
    reused_count: int = 0
    invalidated_count: int = 0
    forced_count: int = 0
    stale_count: int = 0
    recovered_interrupted: int = 0
    temp_files_cleaned: int = 0
    ledger_repaired_tail: bool = False
    ledger_path: Optional[str] = None
    run_id: Optional[str] = None
    mode: str = 'normal'

    def to_dict(self) -> Dict[str, Any]:
        return {'summary': {'total': self.total, 'succeeded': self.succeeded, 'failed': self.failed, 'skipped': self.skipped, 'newly_completed': self.newly_completed, 'reused': self.reused_count, 'invalidated': self.invalidated_count, 'forced': self.forced_count, 'stale': self.stale_count, 'recovered_interrupted': self.recovered_interrupted, 'temp_files_cleaned': self.temp_files_cleaned, 'ledger_repaired_tail': self.ledger_repaired_tail, 'total_duration_ms': round(self.total_duration_ms, 2)}, 'run': {'run_id': self.run_id, 'mode': self.mode, 'ledger': self.ledger_path}, 'config': {'pipeline_file': self.pipeline_config_file, 'input_dir': self.input_dir, 'output_dir': self.output_dir}, 'results': [{'input': r.input_path, 'output': r.output_path, 'outputs': list(r.output_paths), 'success': r.success, 'disposition': r.disposition, 'invalidation_reason': r.invalidation_reason, 'forced': r.forced, 'reused_from_run': r.reused_from_run, 'attempted': r.attempted, 'duration_ms': round(r.duration_ms, 2), 'error': r.error, 'nodes': [{'node_id': nr.node_id, 'node_type': nr.node_type, 'success': nr.success, 'duration_ms': round(nr.duration_ms, 2), 'error': nr.error, 'output_size': list(nr.output_size) if nr.output_size else None} for nr in r.node_results]} for r in self.results]}
