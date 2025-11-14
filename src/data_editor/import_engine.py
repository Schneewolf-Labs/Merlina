"""
Import Engine for Data Editor

Supports multiple file formats and provides automatic schema detection.
Formats: JSON, JSONL, CSV, Parquet, Excel
"""

import json
import csv
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from io import StringIO, BytesIO
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ImportEngine:
    """Engine for importing datasets from various file formats"""

    SUPPORTED_FORMATS = {
        '.json': 'json',
        '.jsonl': 'jsonl',
        '.csv': 'csv',
        '.tsv': 'tsv',
        '.parquet': 'parquet',
        '.xlsx': 'excel',
        '.xls': 'excel'
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_format(self, filename: str) -> Optional[str]:
        """Detect file format from filename extension"""
        ext = Path(filename).suffix.lower()
        return self.SUPPORTED_FORMATS.get(ext)

    def import_file(self, file_path: str, file_content: Optional[bytes] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Import file and return list of rows and metadata

        Args:
            file_path: Path to file or filename for format detection
            file_content: Optional file content as bytes (for uploaded files)

        Returns:
            Tuple of (rows, metadata)
        """
        format_type = self.detect_format(file_path)
        if not format_type:
            raise ValueError(f"Unsupported file format: {file_path}")

        self.logger.info(f"Importing file: {file_path} (format: {format_type})")

        # Read file content
        if file_content is None:
            with open(file_path, 'rb') as f:
                file_content = f.read()

        # Parse based on format
        if format_type == 'json':
            rows = self._parse_json(file_content)
        elif format_type == 'jsonl':
            rows = self._parse_jsonl(file_content)
        elif format_type == 'csv':
            rows = self._parse_csv(file_content, delimiter=',')
        elif format_type == 'tsv':
            rows = self._parse_csv(file_content, delimiter='\t')
        elif format_type == 'parquet':
            rows = self._parse_parquet(file_content)
        elif format_type == 'excel':
            rows = self._parse_excel(file_content)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Generate metadata
        metadata = self._generate_metadata(rows, file_path, format_type)

        self.logger.info(f"Imported {len(rows)} rows from {file_path}")
        return rows, metadata

    def _parse_json(self, content: bytes) -> List[Dict[str, Any]]:
        """Parse JSON file"""
        data = json.loads(content.decode('utf-8'))

        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Check for common nested structures
            if 'data' in data and isinstance(data['data'], list):
                return data['data']
            elif 'rows' in data and isinstance(data['rows'], list):
                return data['rows']
            elif 'examples' in data and isinstance(data['examples'], list):
                return data['examples']
            else:
                # Single object, wrap in list
                return [data]
        else:
            raise ValueError("JSON file must contain a list or object")

    def _parse_jsonl(self, content: bytes) -> List[Dict[str, Any]]:
        """Parse JSONL file (one JSON object per line)"""
        rows = []
        lines = content.decode('utf-8').strip().split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                self.logger.warning(f"Skipping invalid JSON on line {i+1}: {e}")

        return rows

    def _parse_csv(self, content: bytes, delimiter: str = ',') -> List[Dict[str, Any]]:
        """Parse CSV/TSV file"""
        text = content.decode('utf-8')
        reader = csv.DictReader(StringIO(text), delimiter=delimiter)
        return list(reader)

    def _parse_parquet(self, content: bytes) -> List[Dict[str, Any]]:
        """Parse Parquet file"""
        try:
            import pyarrow.parquet as pq
            from io import BytesIO

            table = pq.read_table(BytesIO(content))
            # Convert to list of dicts
            df = table.to_pandas()
            return df.to_dict('records')
        except ImportError:
            raise ImportError("pyarrow is required for Parquet support. Install with: pip install pyarrow")

    def _parse_excel(self, content: bytes) -> List[Dict[str, Any]]:
        """Parse Excel file"""
        try:
            import pandas as pd
            from io import BytesIO

            df = pd.read_excel(BytesIO(content))
            return df.to_dict('records')
        except ImportError:
            raise ImportError("openpyxl is required for Excel support. Install with: pip install openpyxl")

    def _generate_metadata(self, rows: List[Dict[str, Any]], file_path: str, format_type: str) -> Dict[str, Any]:
        """Generate metadata about the imported dataset"""
        if not rows:
            return {
                'file_path': file_path,
                'format': format_type,
                'num_rows': 0,
                'columns': [],
                'sample': None,
                'imported_at': datetime.now().isoformat()
            }

        # Detect all columns across all rows (some rows may have different keys)
        all_columns = set()
        for row in rows:
            all_columns.update(row.keys())

        # Column statistics
        column_stats = {}
        for col in all_columns:
            non_null_count = sum(1 for row in rows if row.get(col) is not None)
            column_stats[col] = {
                'non_null_count': non_null_count,
                'null_count': len(rows) - non_null_count,
                'fill_rate': non_null_count / len(rows) if rows else 0
            }

        return {
            'file_path': file_path,
            'format': format_type,
            'num_rows': len(rows),
            'columns': sorted(all_columns),
            'column_stats': column_stats,
            'sample': rows[0] if rows else None,
            'imported_at': datetime.now().isoformat()
        }

    def detect_schema_type(self, rows: List[Dict[str, Any]]) -> str:
        """
        Detect common dataset schema types

        Returns: 'orpo', 'sharegpt', 'alpaca', 'completion', 'messages', 'unknown'
        """
        if not rows:
            return 'unknown'

        first_row = rows[0]
        columns = set(first_row.keys())

        # Check for ORPO format
        if {'prompt', 'chosen', 'rejected'}.issubset(columns):
            return 'orpo'

        # Check for ShareGPT format
        if 'conversations' in columns or 'messages' in columns:
            return 'sharegpt'

        # Check for Alpaca format
        if 'instruction' in columns and 'output' in columns:
            return 'alpaca'

        # Check for OpenAI messages format
        if 'messages' in columns:
            messages = first_row.get('messages')
            if isinstance(messages, list) and messages and isinstance(messages[0], dict):
                if 'role' in messages[0] and 'content' in messages[0]:
                    return 'messages'

        # Check for simple completion format
        if 'prompt' in columns and 'completion' in columns:
            return 'completion'

        # Check for question-answer format
        if ('question' in columns or 'input' in columns) and \
           ('answer' in columns or 'output' in columns or 'response' in columns):
            return 'qa'

        return 'unknown'

    def suggest_column_mapping(self, rows: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        """
        Suggest column mapping based on detected schema

        Returns mapping for: prompt, chosen, rejected, system, reasoning
        """
        if not rows:
            return {'prompt': None, 'chosen': None, 'rejected': None, 'system': None, 'reasoning': None}

        schema_type = self.detect_schema_type(rows)
        first_row = rows[0]
        columns = set(first_row.keys())

        mapping = {
            'prompt': None,
            'chosen': None,
            'rejected': None,
            'system': None,
            'reasoning': None
        }

        if schema_type == 'orpo':
            # Already in ORPO format
            if 'prompt' in columns:
                mapping['prompt'] = 'prompt'
            if 'chosen' in columns:
                mapping['chosen'] = 'chosen'
            if 'rejected' in columns:
                mapping['rejected'] = 'rejected'
            if 'system' in columns:
                mapping['system'] = 'system'
            if 'reasoning' in columns:
                mapping['reasoning'] = 'reasoning'

        elif schema_type == 'alpaca':
            # Alpaca: instruction -> prompt, output -> chosen
            mapping['prompt'] = 'instruction'
            mapping['chosen'] = 'output'
            if 'input' in columns:
                mapping['prompt'] = '{instruction}\n{input}'  # Template
            # Note: rejected needs to be generated

        elif schema_type == 'completion':
            mapping['prompt'] = 'prompt'
            mapping['chosen'] = 'completion'

        elif schema_type == 'qa':
            # Question-answer format
            if 'question' in columns:
                mapping['prompt'] = 'question'
            elif 'input' in columns:
                mapping['prompt'] = 'input'

            if 'answer' in columns:
                mapping['chosen'] = 'answer'
            elif 'response' in columns:
                mapping['chosen'] = 'response'
            elif 'output' in columns:
                mapping['chosen'] = 'output'

        elif schema_type in ['sharegpt', 'messages']:
            # These need special handling - extract from conversations/messages
            mapping['prompt'] = 'messages'  # Needs extraction logic
            mapping['chosen'] = 'messages'  # Needs extraction logic

        else:
            # Unknown - try to guess from column names
            prompt_keywords = ['prompt', 'question', 'input', 'instruction', 'query', 'user']
            chosen_keywords = ['chosen', 'response', 'answer', 'output', 'completion', 'assistant']
            rejected_keywords = ['rejected', 'bad_response', 'negative']
            system_keywords = ['system', 'system_prompt', 'context']

            for col in columns:
                col_lower = col.lower()
                if mapping['prompt'] is None and any(kw in col_lower for kw in prompt_keywords):
                    mapping['prompt'] = col
                elif mapping['chosen'] is None and any(kw in col_lower for kw in chosen_keywords):
                    mapping['chosen'] = col
                elif mapping['rejected'] is None and any(kw in col_lower for kw in rejected_keywords):
                    mapping['rejected'] = col
                elif mapping['system'] is None and any(kw in col_lower for kw in system_keywords):
                    mapping['system'] = col

        return mapping

    def extract_nested_value(self, row: Dict[str, Any], path: str) -> Any:
        """
        Extract value from nested dict using dot notation or array indices

        Examples:
            "messages[0].content"
            "data.response"
            "conversations[1].value"
        """
        parts = self._parse_path(path)
        value = row

        for part in parts:
            if isinstance(part, int):
                # Array index
                if isinstance(value, list) and 0 <= part < len(value):
                    value = value[part]
                else:
                    return None
            else:
                # Dict key
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None

            if value is None:
                return None

        return value

    def _parse_path(self, path: str) -> List[Any]:
        """
        Parse path string into list of keys and indices

        "messages[0].content" -> ["messages", 0, "content"]
        """
        import re
        parts = []

        # Split by dots, but handle array indices
        tokens = re.split(r'\.', path)

        for token in tokens:
            # Check if token has array index
            match = re.match(r'(\w+)\[(\d+)\]', token)
            if match:
                parts.append(match.group(1))
                parts.append(int(match.group(2)))
            else:
                parts.append(token)

        return parts


# Convenience function
def import_dataset(file_path: str, file_content: Optional[bytes] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience function to import a dataset

    Returns: (rows, metadata)
    """
    engine = ImportEngine()
    return engine.import_file(file_path, file_content)
