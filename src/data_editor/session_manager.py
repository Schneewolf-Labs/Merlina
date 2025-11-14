"""
Session Manager for Data Editor

Provides SQLite-based persistence for editing sessions with undo/redo support.
"""

import sqlite3
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from contextlib import contextmanager

from . import EditorSession, EditorRow, OperationType

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages editor sessions with SQLite persistence and undo/redo support
    """

    def __init__(self, db_path: str = "./data/editor_sessions.db"):
        self.db_path = db_path
        self._ensure_database()
        self.logger = logging.getLogger(__name__)

    def _ensure_database(self):
        """Create database and tables if they don't exist"""
        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_file TEXT,
                    column_mapping TEXT,
                    statistics TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    num_rows INTEGER DEFAULT 0
                )
            """)

            # Rows table (stores current state)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rows (
                    session_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    prompt TEXT,
                    chosen TEXT,
                    rejected TEXT,
                    system TEXT,
                    reasoning TEXT,
                    metadata TEXT,
                    validation_errors TEXT,
                    validation_warnings TEXT,
                    PRIMARY KEY (session_id, idx),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)

            # Operations history for undo/redo
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    before_state TEXT,
                    after_state TEXT,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)

            # Undo/redo pointers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS undo_state (
                    session_id TEXT PRIMARY KEY,
                    current_operation_id INTEGER,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rows_session ON rows(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_session ON operations(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_timestamp ON operations(timestamp)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def create_session(self, name: str, source_file: Optional[str] = None) -> str:
        """
        Create a new editing session

        Returns: session_id
        """
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, name, source_file, created_at, updated_at, num_rows)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (session_id, name, source_file, now, now))

            # Initialize undo state
            cursor.execute("""
                INSERT INTO undo_state (session_id, current_operation_id)
                VALUES (?, NULL)
            """, (session_id,))

            conn.commit()

        self.logger.info(f"Created session: {session_id} ({name})")
        return session_id

    def get_session(self, session_id: str) -> Optional[EditorSession]:
        """Get a session by ID with all its rows"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get session metadata
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            session_row = cursor.fetchone()

            if not session_row:
                return None

            # Get all rows
            cursor.execute("""
                SELECT * FROM rows WHERE session_id = ? ORDER BY idx
            """, (session_id,))

            rows = []
            for row in cursor.fetchall():
                editor_row = EditorRow(
                    idx=row['idx'],
                    prompt=row['prompt'],
                    chosen=row['chosen'],
                    rejected=row['rejected'],
                    system=row['system'],
                    reasoning=row['reasoning'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    validation_errors=json.loads(row['validation_errors']) if row['validation_errors'] else [],
                    validation_warnings=json.loads(row['validation_warnings']) if row['validation_warnings'] else []
                )
                rows.append(editor_row)

            return EditorSession(
                session_id=session_row['session_id'],
                name=session_row['name'],
                rows=rows,
                created_at=datetime.fromisoformat(session_row['created_at']),
                updated_at=datetime.fromisoformat(session_row['updated_at']),
                source_file=session_row['source_file'],
                column_mapping=json.loads(session_row['column_mapping']) if session_row['column_mapping'] else None,
                statistics=json.loads(session_row['statistics']) if session_row['statistics'] else {}
            )

    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session metadata"""
        allowed_fields = ['name', 'source_file', 'column_mapping', 'statistics']
        updates = []
        values = []

        for key, value in kwargs.items():
            if key in allowed_fields:
                if key in ['column_mapping', 'statistics']:
                    value = json.dumps(value)
                updates.append(f"{key} = ?")
                values.append(value)

        if not updates:
            return False

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(session_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE sessions
                SET {', '.join(updates)}
                WHERE session_id = ?
            """, values)
            conn.commit()

        return cursor.rowcount > 0

    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List all sessions with metadata"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, name, source_file, num_rows, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

            sessions = []
            for row in cursor.fetchall():
                sessions.append(dict(row))

            return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all associated data"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

        return cursor.rowcount > 0

    def add_rows(self, session_id: str, rows: List[EditorRow], record_operation: bool = True) -> bool:
        """Add multiple rows to a session"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Record operation for undo if requested
            if record_operation:
                before_state = None
                after_state = json.dumps([row.to_dict() for row in rows])
                self._record_operation(cursor, session_id, OperationType.ADD_ROW, before_state, after_state)

            # Insert rows
            for row in rows:
                cursor.execute("""
                    INSERT OR REPLACE INTO rows
                    (session_id, idx, prompt, chosen, rejected, system, reasoning, metadata, validation_errors, validation_warnings)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    row.idx,
                    row.prompt,
                    row.chosen,
                    row.rejected,
                    row.system,
                    row.reasoning,
                    json.dumps(row.metadata),
                    json.dumps(row.validation_errors),
                    json.dumps(row.validation_warnings)
                ))

            # Update row count
            cursor.execute("SELECT COUNT(*) FROM rows WHERE session_id = ?", (session_id,))
            num_rows = cursor.fetchone()[0]

            cursor.execute("""
                UPDATE sessions
                SET num_rows = ?, updated_at = ?
                WHERE session_id = ?
            """, (num_rows, datetime.now().isoformat(), session_id))

            conn.commit()

        return True

    def update_row(self, session_id: str, idx: int, updates: Dict[str, Any], record_operation: bool = True) -> bool:
        """Update a specific row"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get current state for undo
            if record_operation:
                cursor.execute("SELECT * FROM rows WHERE session_id = ? AND idx = ?", (session_id, idx))
                current = cursor.fetchone()
                if current:
                    before_state = json.dumps(dict(current))
                else:
                    before_state = None

            # Build update query
            allowed_fields = ['prompt', 'chosen', 'rejected', 'system', 'reasoning', 'metadata', 'validation_errors', 'validation_warnings']
            set_clauses = []
            values = []

            for key, value in updates.items():
                if key in allowed_fields:
                    if key in ['metadata', 'validation_errors', 'validation_warnings'] and not isinstance(value, str):
                        value = json.dumps(value)
                    set_clauses.append(f"{key} = ?")
                    values.append(value)

            if not set_clauses:
                return False

            values.extend([session_id, idx])

            cursor.execute(f"""
                UPDATE rows
                SET {', '.join(set_clauses)}
                WHERE session_id = ? AND idx = ?
            """, values)

            # Record operation for undo
            if record_operation and cursor.rowcount > 0:
                cursor.execute("SELECT * FROM rows WHERE session_id = ? AND idx = ?", (session_id, idx))
                updated = cursor.fetchone()
                after_state = json.dumps(dict(updated))
                self._record_operation(cursor, session_id, OperationType.UPDATE_ROW, before_state, after_state, {"idx": idx})

            # Update session timestamp
            cursor.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                          (datetime.now().isoformat(), session_id))

            conn.commit()

        return cursor.rowcount > 0

    def delete_row(self, session_id: str, idx: int, record_operation: bool = True) -> bool:
        """Delete a specific row"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get current state for undo
            if record_operation:
                cursor.execute("SELECT * FROM rows WHERE session_id = ? AND idx = ?", (session_id, idx))
                current = cursor.fetchone()
                before_state = json.dumps(dict(current)) if current else None

            cursor.execute("DELETE FROM rows WHERE session_id = ? AND idx = ?", (session_id, idx))

            # Record operation for undo
            if record_operation and cursor.rowcount > 0:
                self._record_operation(cursor, session_id, OperationType.DELETE_ROW, before_state, None, {"idx": idx})

            # Update row count
            cursor.execute("SELECT COUNT(*) FROM rows WHERE session_id = ?", (session_id,))
            num_rows = cursor.fetchone()[0]

            cursor.execute("""
                UPDATE sessions
                SET num_rows = ?, updated_at = ?
                WHERE session_id = ?
            """, (num_rows, datetime.now().isoformat(), session_id))

            conn.commit()

        return cursor.rowcount > 0

    def _record_operation(self, cursor, session_id: str, operation_type: OperationType,
                         before_state: Optional[str], after_state: Optional[str],
                         metadata: Optional[Dict[str, Any]] = None):
        """Record an operation for undo/redo"""
        cursor.execute("""
            INSERT INTO operations (session_id, operation_type, timestamp, before_state, after_state, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            operation_type.value,
            datetime.now().isoformat(),
            before_state,
            after_state,
            json.dumps(metadata) if metadata else None
        ))

        # Update undo state pointer to latest operation
        operation_id = cursor.lastrowid
        cursor.execute("""
            UPDATE undo_state
            SET current_operation_id = ?
            WHERE session_id = ?
        """, (operation_id, session_id))

    def undo(self, session_id: str) -> bool:
        """Undo the last operation"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get current operation pointer
            cursor.execute("SELECT current_operation_id FROM undo_state WHERE session_id = ?", (session_id,))
            result = cursor.fetchone()

            if not result or not result['current_operation_id']:
                return False  # Nothing to undo

            current_op_id = result['current_operation_id']

            # Get the operation to undo
            cursor.execute("""
                SELECT * FROM operations WHERE id = ? AND session_id = ?
            """, (current_op_id, session_id))

            operation = cursor.fetchone()
            if not operation:
                return False

            # Apply the undo (restore before_state)
            success = self._apply_undo(cursor, operation)

            if success:
                # Move pointer back
                cursor.execute("""
                    SELECT id FROM operations
                    WHERE session_id = ? AND id < ?
                    ORDER BY id DESC
                    LIMIT 1
                """, (session_id, current_op_id))

                prev_op = cursor.fetchone()
                new_pointer = prev_op['id'] if prev_op else None

                cursor.execute("""
                    UPDATE undo_state
                    SET current_operation_id = ?
                    WHERE session_id = ?
                """, (new_pointer, session_id))

                conn.commit()

            return success

    def redo(self, session_id: str) -> bool:
        """Redo the next operation"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get current operation pointer
            cursor.execute("SELECT current_operation_id FROM undo_state WHERE session_id = ?", (session_id,))
            result = cursor.fetchone()

            current_op_id = result['current_operation_id'] if result else None

            # Get the next operation to redo
            if current_op_id:
                cursor.execute("""
                    SELECT * FROM operations
                    WHERE session_id = ? AND id > ?
                    ORDER BY id ASC
                    LIMIT 1
                """, (session_id, current_op_id))
            else:
                cursor.execute("""
                    SELECT * FROM operations
                    WHERE session_id = ?
                    ORDER BY id ASC
                    LIMIT 1
                """, (session_id,))

            operation = cursor.fetchone()
            if not operation:
                return False  # Nothing to redo

            # Apply the redo (restore after_state)
            success = self._apply_redo(cursor, operation)

            if success:
                cursor.execute("""
                    UPDATE undo_state
                    SET current_operation_id = ?
                    WHERE session_id = ?
                """, (operation['id'], session_id))

                conn.commit()

            return success

    def _apply_undo(self, cursor, operation) -> bool:
        """Apply undo operation"""
        op_type = OperationType(operation['operation_type'])
        session_id = operation['session_id']
        before_state = operation['before_state']

        if op_type == OperationType.ADD_ROW:
            # Undo add: delete the rows
            rows_data = json.loads(operation['after_state'])
            for row_data in rows_data:
                cursor.execute("DELETE FROM rows WHERE session_id = ? AND idx = ?",
                             (session_id, row_data['idx']))

        elif op_type == OperationType.UPDATE_ROW:
            # Undo update: restore previous state
            if before_state:
                row_data = json.loads(before_state)
                cursor.execute("""
                    UPDATE rows
                    SET prompt=?, chosen=?, rejected=?, system=?, reasoning=?, metadata=?, validation_errors=?, validation_warnings=?
                    WHERE session_id = ? AND idx = ?
                """, (
                    row_data['prompt'], row_data['chosen'], row_data['rejected'],
                    row_data['system'], row_data['reasoning'], row_data['metadata'],
                    row_data['validation_errors'], row_data['validation_warnings'],
                    session_id, row_data['idx']
                ))

        elif op_type == OperationType.DELETE_ROW:
            # Undo delete: restore the row
            if before_state:
                row_data = json.loads(before_state)
                cursor.execute("""
                    INSERT INTO rows (session_id, idx, prompt, chosen, rejected, system, reasoning, metadata, validation_errors, validation_warnings)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, row_data['idx'],
                    row_data['prompt'], row_data['chosen'], row_data['rejected'],
                    row_data['system'], row_data['reasoning'], row_data['metadata'],
                    row_data['validation_errors'], row_data['validation_warnings']
                ))

        return True

    def _apply_redo(self, cursor, operation) -> bool:
        """Apply redo operation"""
        op_type = OperationType(operation['operation_type'])
        session_id = operation['session_id']
        after_state = operation['after_state']

        if op_type == OperationType.ADD_ROW:
            # Redo add: insert the rows again
            if after_state:
                rows_data = json.loads(after_state)
                for row_data in rows_data:
                    cursor.execute("""
                        INSERT OR REPLACE INTO rows (session_id, idx, prompt, chosen, rejected, system, reasoning, metadata, validation_errors, validation_warnings)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session_id, row_data['idx'],
                        row_data['prompt'], row_data['chosen'], row_data['rejected'],
                        row_data['system'], row_data['reasoning'],
                        json.dumps(row_data['metadata']), json.dumps(row_data['validation_errors']),
                        json.dumps(row_data['validation_warnings'])
                    ))

        elif op_type == OperationType.UPDATE_ROW:
            # Redo update: apply the new state
            if after_state:
                row_data = json.loads(after_state)
                cursor.execute("""
                    UPDATE rows
                    SET prompt=?, chosen=?, rejected=?, system=?, reasoning=?, metadata=?, validation_errors=?, validation_warnings=?
                    WHERE session_id = ? AND idx = ?
                """, (
                    row_data['prompt'], row_data['chosen'], row_data['rejected'],
                    row_data['system'], row_data['reasoning'], row_data['metadata'],
                    row_data['validation_errors'], row_data['validation_warnings'],
                    session_id, row_data['idx']
                ))

        elif op_type == OperationType.DELETE_ROW:
            # Redo delete: delete the row again
            metadata = json.loads(operation['metadata']) if operation['metadata'] else {}
            idx = metadata.get('idx')
            if idx is not None:
                cursor.execute("DELETE FROM rows WHERE session_id = ? AND idx = ?",
                             (session_id, idx))

        return True

    def cleanup_old_sessions(self, days: int = 7) -> int:
        """Delete sessions older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM sessions
                WHERE updated_at < ?
            """, (cutoff.isoformat(),))
            conn.commit()

        deleted = cursor.rowcount
        self.logger.info(f"Cleaned up {deleted} old sessions")
        return deleted
