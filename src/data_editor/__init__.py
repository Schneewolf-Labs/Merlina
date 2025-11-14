"""
Data Editor Module for Merlina

This module provides comprehensive data editing capabilities for transforming
raw datasets into ORPO-ready format with proper schema validation.

Components:
- import_engine: Multi-format file importing (JSON, JSONL, CSV, Parquet)
- session_manager: Session persistence with undo/redo support
- validation: Schema and quality validation engine
- transformations: Data transformation and column mapping
- pair_generators: Preference pair generation strategies
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

__version__ = "1.0.0"


class OperationType(Enum):
    """Types of operations that can be performed in the editor"""
    IMPORT = "import"
    ADD_ROW = "add_row"
    UPDATE_ROW = "update_row"
    DELETE_ROW = "delete_row"
    TRANSFORM = "transform"
    MAP_COLUMNS = "map_columns"
    GENERATE_PAIRS = "generate_pairs"
    VALIDATE = "validate"


@dataclass
class EditorRow:
    """Represents a single row in the dataset editor"""
    idx: int
    prompt: Optional[str] = None
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    system: Optional[str] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "idx": self.idx,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "system": self.system,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditorRow":
        """Create from dictionary"""
        return cls(
            idx=data["idx"],
            prompt=data.get("prompt"),
            chosen=data.get("chosen"),
            rejected=data.get("rejected"),
            system=data.get("system"),
            reasoning=data.get("reasoning"),
            metadata=data.get("metadata", {}),
            validation_errors=data.get("validation_errors", []),
            validation_warnings=data.get("validation_warnings", [])
        )

    def get_required_fields(self) -> Dict[str, Optional[str]]:
        """Get only the required ORPO fields"""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected
        }

    def get_all_fields(self) -> Dict[str, Optional[str]]:
        """Get all ORPO fields including optional ones"""
        result = self.get_required_fields()
        if self.system is not None:
            result["system"] = self.system
        if self.reasoning is not None:
            result["reasoning"] = self.reasoning
        return result


@dataclass
class EditorSession:
    """Represents an editing session"""
    session_id: str
    name: str
    rows: List[EditorRow]
    created_at: datetime
    updated_at: datetime
    source_file: Optional[str] = None
    column_mapping: Optional[Dict[str, Any]] = None
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "rows": [row.to_dict() for row in self.rows],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_file": self.source_file,
            "column_mapping": self.column_mapping,
            "statistics": self.statistics
        }

    def get_row(self, idx: int) -> Optional[EditorRow]:
        """Get row by index"""
        for row in self.rows:
            if row.idx == idx:
                return row
        return None

    def add_row(self, row: EditorRow) -> None:
        """Add a new row"""
        self.rows.append(row)
        self.updated_at = datetime.now()

    def update_row(self, idx: int, updates: Dict[str, Any]) -> bool:
        """Update a row by index"""
        row = self.get_row(idx)
        if row is None:
            return False

        for key, value in updates.items():
            if hasattr(row, key):
                setattr(row, key, value)

        self.updated_at = datetime.now()
        return True

    def delete_row(self, idx: int) -> bool:
        """Delete a row by index"""
        row = self.get_row(idx)
        if row is None:
            return False

        self.rows.remove(row)
        self.updated_at = datetime.now()
        return True

    def get_valid_rows(self) -> List[EditorRow]:
        """Get only rows without validation errors"""
        return [row for row in self.rows if not row.validation_errors]


@dataclass
class ValidationResult:
    """Result of validation checks"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    row_issues: Dict[int, Dict[str, List[str]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.statistics,
            "row_issues": self.row_issues
        }


@dataclass
class TransformationConfig:
    """Configuration for data transformation"""
    column_mapping: Dict[str, Any]
    generate_rejected: bool = False
    rejected_strategy: Optional[str] = None
    add_system_message: Optional[str] = None
    extract_reasoning: bool = False
    custom_transforms: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "column_mapping": self.column_mapping,
            "generate_rejected": self.generate_rejected,
            "rejected_strategy": self.rejected_strategy,
            "add_system_message": self.add_system_message,
            "extract_reasoning": self.extract_reasoning,
            "custom_transforms": self.custom_transforms
        }


# Export main classes
__all__ = [
    "OperationType",
    "EditorRow",
    "EditorSession",
    "ValidationResult",
    "TransformationConfig"
]
