"""
Custom exception classes for Merlina training system.

This module provides a hierarchy of exceptions that replace fragile
string matching with proper exception types, enabling cleaner error
handling throughout the application.
"""

from typing import Optional, Any


class MerlinaError(Exception):
    """Base exception for all Merlina-related errors."""

    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


# =============================================================================
# Dataset Errors
# =============================================================================

class DatasetError(MerlinaError):
    """Base exception for dataset-related errors."""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset cannot be found at the specified source."""

    def __init__(self, source: str, source_type: str = "unknown"):
        self.source = source
        self.source_type = source_type
        message = f"Dataset not found: '{source}' (source type: {source_type})"
        super().__init__(message, details={"source": source, "source_type": source_type})


class DatasetLoadError(DatasetError):
    """Raised when a dataset fails to load."""

    def __init__(self, source: str, reason: str):
        self.source = source
        self.reason = reason
        message = f"Failed to load dataset '{source}': {reason}"
        super().__init__(message, details={"source": source, "reason": reason})


class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails."""

    def __init__(self, message: str, missing_columns: Optional[list] = None):
        self.missing_columns = missing_columns
        super().__init__(message, details={"missing_columns": missing_columns})


class ColumnMappingError(DatasetError):
    """Raised when column mapping fails."""

    def __init__(self, source_column: str, available_columns: list):
        self.source_column = source_column
        self.available_columns = available_columns
        message = (
            f"Column '{source_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
        super().__init__(message, details={
            "source_column": source_column,
            "available_columns": available_columns
        })


class InvalidDatasetFormatError(DatasetError):
    """Raised when the dataset format is invalid or unsupported."""

    def __init__(self, format_name: str, supported_formats: list):
        self.format_name = format_name
        self.supported_formats = supported_formats
        message = (
            f"Invalid dataset format: '{format_name}'. "
            f"Supported formats: {supported_formats}"
        )
        super().__init__(message, details={
            "format_name": format_name,
            "supported_formats": supported_formats
        })


# =============================================================================
# Model Errors
# =============================================================================

class ModelError(MerlinaError):
    """Base exception for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a model cannot be found."""

    def __init__(self, model_path: str, is_local: bool = False):
        self.model_path = model_path
        self.is_local = is_local
        source = "local path" if is_local else "HuggingFace Hub"
        message = f"Model not found at {source}: '{model_path}'"
        super().__init__(message, details={"model_path": model_path, "is_local": is_local})


class ModelAccessError(ModelError):
    """Raised when model access is denied (e.g., gated model without token)."""

    def __init__(self, model_path: str, reason: str):
        self.model_path = model_path
        self.reason = reason
        message = f"Cannot access model '{model_path}': {reason}"
        super().__init__(message, details={"model_path": model_path, "reason": reason})


class ModelLoadError(ModelError):
    """Raised when model loading fails."""

    def __init__(self, model_path: str, reason: str):
        self.model_path = model_path
        self.reason = reason
        message = f"Failed to load model '{model_path}': {reason}"
        super().__init__(message, details={"model_path": model_path, "reason": reason})


# =============================================================================
# Training Errors
# =============================================================================

class TrainingError(MerlinaError):
    """Base exception for training-related errors."""
    pass


class TrainingConfigError(TrainingError):
    """Raised when training configuration is invalid."""

    def __init__(self, field: str, value: Any, reason: str):
        self.field = field
        self.value = value
        self.reason = reason
        message = f"Invalid training config for '{field}': {reason} (value: {value})"
        super().__init__(message, details={"field": field, "value": value, "reason": reason})


class InsufficientResourcesError(TrainingError):
    """Raised when system resources are insufficient for training."""

    def __init__(self, resource: str, required: Any, available: Any):
        self.resource = resource
        self.required = required
        self.available = available
        message = (
            f"Insufficient {resource}: required {required}, "
            f"but only {available} available"
        )
        super().__init__(message, details={
            "resource": resource,
            "required": required,
            "available": available
        })


class TrainingInterruptedError(TrainingError):
    """Raised when training is interrupted (e.g., by stop request)."""

    def __init__(self, job_id: str, step: Optional[int] = None):
        self.job_id = job_id
        self.step = step
        message = f"Training interrupted for job {job_id}"
        if step is not None:
            message += f" at step {step}"
        super().__init__(message, details={"job_id": job_id, "step": step})


# =============================================================================
# Job Queue Errors
# =============================================================================

class JobError(MerlinaError):
    """Base exception for job-related errors."""
    pass


class JobNotFoundError(JobError):
    """Raised when a job cannot be found."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        message = f"Job not found: {job_id}"
        super().__init__(message, details={"job_id": job_id})


class JobAlreadyExistsError(JobError):
    """Raised when attempting to create a job with an existing ID."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        message = f"Job already exists: {job_id}"
        super().__init__(message, details={"job_id": job_id})


class JobQueueFullError(JobError):
    """Raised when the job queue is full."""

    def __init__(self, max_queue_size: int):
        self.max_queue_size = max_queue_size
        message = f"Job queue is full (max size: {max_queue_size})"
        super().__init__(message, details={"max_queue_size": max_queue_size})


class JobCancellationError(JobError):
    """Raised when job cancellation fails."""

    def __init__(self, job_id: str, reason: str):
        self.job_id = job_id
        self.reason = reason
        message = f"Failed to cancel job {job_id}: {reason}"
        super().__init__(message, details={"job_id": job_id, "reason": reason})


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(MerlinaError):
    """Base exception for validation errors."""
    pass


class PreflightValidationError(ValidationError):
    """Raised when preflight validation fails."""

    def __init__(self, errors: list, warnings: Optional[list] = None):
        self.errors = errors
        self.warnings = warnings or []
        message = f"Preflight validation failed with {len(errors)} error(s)"
        super().__init__(message, details={"errors": errors, "warnings": self.warnings})


# =============================================================================
# Upload Errors
# =============================================================================

class UploadError(MerlinaError):
    """Base exception for upload-related errors."""
    pass


class HuggingFaceUploadError(UploadError):
    """Raised when HuggingFace Hub upload fails."""

    def __init__(self, repo_id: str, reason: str):
        self.repo_id = repo_id
        self.reason = reason
        message = f"Failed to upload to HuggingFace Hub '{repo_id}': {reason}"
        super().__init__(message, details={"repo_id": repo_id, "reason": reason})


class FileUploadError(UploadError):
    """Raised when file upload fails."""

    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        message = f"Failed to upload file '{filename}': {reason}"
        super().__init__(message, details={"filename": filename, "reason": reason})
