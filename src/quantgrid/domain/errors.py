"""Error taxonomy shared by the API, workers, coordinator and agent.

API responses use the envelope ``{"error": {"type": <ErrorCode>, "message": ..., ...}}``.
Task-execution failures are recorded per attempt with a specific ``ErrorCode`` and are
surfaced through the API under the umbrella type ``WORKER_FAILURE``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal


class ErrorCode(StrEnum):
    # --- request / API level -------------------------------------------------------
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    CONFLICT = "CONFLICT"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    UNAUTHORIZED = "UNAUTHORIZED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    PLANNER_ERROR = "PLANNER_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    # --- task execution (attempt.error_code); API umbrella type WORKER_FAILURE -------
    WORKER_FAILURE = "WORKER_FAILURE"
    WORKER_LOST = "WORKER_LOST"  # lease expired: worker crashed, was killed or stalled
    TASK_TIMEOUT = "TASK_TIMEOUT"
    TRANSIENT_ERROR = "TRANSIENT_ERROR"
    TASK_ERROR = "TASK_ERROR"  # unexpected exception inside task code
    PERMANENT_TASK_ERROR = "PERMANENT_TASK_ERROR"  # task code declared the failure final
    DATA_INTEGRITY_ERROR = "DATA_INTEGRITY_ERROR"  # dataset missing / checksum mismatch
    SPEC_ERROR = "SPEC_ERROR"  # immutable spec failed re-validation at execution time
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    ARTIFACT_ERROR = "ARTIFACT_ERROR"


#: Error codes for which an automatic retry (up to ``max_attempts``) is sensible.
RETRYABLE_TASK_ERRORS: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.WORKER_LOST,
        ErrorCode.TASK_TIMEOUT,
        ErrorCode.TRANSIENT_ERROR,
        ErrorCode.TASK_ERROR,
        ErrorCode.ARTIFACT_ERROR,
    }
)


class QuantGridError(Exception):
    """Base class for all expected (typed) failures."""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    http_status: int = 500
    retryable: bool = False

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class ValidationFailed(QuantGridError):
    code = ErrorCode.VALIDATION_ERROR
    http_status = 422


class QuotaExceeded(QuantGridError):
    code = ErrorCode.QUOTA_EXCEEDED
    http_status = 422


class NotFound(QuantGridError):
    code = ErrorCode.NOT_FOUND
    http_status = 404


class Conflict(QuantGridError):
    code = ErrorCode.CONFLICT
    http_status = 409


class RateLimited(QuantGridError):
    code = ErrorCode.RATE_LIMITED
    http_status = 429

    def __init__(self, message: str, *, retry_after_s: int, details: Any = None) -> None:
        super().__init__(message, details=details)
        self.retry_after_s = retry_after_s


class Unauthorized(QuantGridError):
    code = ErrorCode.UNAUTHORIZED
    http_status = 401


class PayloadTooLarge(QuantGridError):
    code = ErrorCode.PAYLOAD_TOO_LARGE
    http_status = 413


class PlannerError(QuantGridError):
    code = ErrorCode.PLANNER_ERROR
    http_status = 502


# --------------------------------------------------------------------------------------
# Task execution errors (raised inside workers / the domain engine)
# --------------------------------------------------------------------------------------


class TaskExecutionError(QuantGridError):
    """A task attempt failed. ``code`` is recorded on the attempt row."""

    code = ErrorCode.TASK_ERROR
    retryable = True


class TransientTaskError(TaskExecutionError):
    code = ErrorCode.TRANSIENT_ERROR
    retryable = True


class TaskTimeout(TaskExecutionError):
    code = ErrorCode.TASK_TIMEOUT
    retryable = True


class DataIntegrityError(TaskExecutionError):
    code = ErrorCode.DATA_INTEGRITY_ERROR
    retryable = False


class SpecError(TaskExecutionError):
    code = ErrorCode.SPEC_ERROR
    retryable = False


class ResourceLimitExceeded(TaskExecutionError):
    code = ErrorCode.RESOURCE_LIMIT
    retryable = False


class ArtifactError(TaskExecutionError):
    code = ErrorCode.ARTIFACT_ERROR
    retryable = True


class PermanentTaskError(TaskExecutionError):
    """Non-retryable task failure (e.g. injected permanent fault). Has its own error code
    so that ``retry-failed`` (which classifies by code) never treats it as retryable."""

    code = ErrorCode.PERMANENT_TASK_ERROR
    retryable = False


AbortReason = Literal["CANCELED", "LEASE_LOST", "SHUTDOWN"]


class TaskAborted(Exception):
    """Cooperative stop signal raised by a cancel token inside long computations.

    * ``CANCELED``   - the batch was cancelled; the worker acknowledges the cancel.
    * ``LEASE_LOST`` - the coordinator reclaimed the lease; the worker must not publish.
    * ``SHUTDOWN``   - the worker process is shutting down; the lease will expire and the
      coordinator re-queues the task.
    """

    def __init__(self, reason: AbortReason) -> None:
        super().__init__(reason)
        self.reason: AbortReason = reason
