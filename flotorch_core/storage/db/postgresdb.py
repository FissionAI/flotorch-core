import psycopg2
from psycopg2.extras import RealDictCursor
from flotorch_core.storage.db.db_storage import DBStorage
from flotorch_core.storage.db.db_utils import _prepare_item_for_write
from typing import List, Dict, Any, Optional
from flotorch_core.logger.global_logger import get_logger

logger = get_logger()

class PostgresDB(DBStorage):
    def __init__(self, dbname: str, user: str, password: str, table_name: str, host: str = "localhost", port: int = 5432):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.table = f'"{table_name}"'
        self.host = host
        self.port = port
        self.conn = self._connect()

    def _connect(self):
        try:
            return psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
        except psycopg2.Error as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            return None

    def write(self, item: dict):
        """
        Insert a single item into the specified table.
        """
        if not self.conn:
            return False
        
        item = _prepare_item_for_write(item)
        columns = ", ".join([f'"{k}"' for k in item.keys()])
        values = ", ".join([f"%({k})s" for k in item.keys()])
        query = f"INSERT INTO {self.table} ({columns}) VALUES ({values}) ON CONFLICT DO NOTHING"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, item)
                self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Error writing to PostgreSQL: {e}")
            return False

    def read(self, key: Optional[Dict[str, Any]] = None) -> Optional[List[dict]]:
            """
            Retrieve item(s) based on key.
            If no key is provided, retrieves all items from the table.
            """
            if not self.conn:
                logger.error("No database connection available.")
                return None

            try:
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if key and len(key) > 0:
                        key_column, key_value = next(iter(key.items()))
                        quoted_key_column = f'"{key_column}"'
                        query = f"SELECT * FROM {self.table} WHERE {quoted_key_column} = %s"
                        cur.execute(query, (key_value,))
                    else:
                        logger.info("No key provided, retrieving all items.")
                        query = f"SELECT * FROM {self.table}"
                        cur.execute(query)
                    
                    result = cur.fetchall()
                    return result if result else []

            except psycopg2.Error as e:
                logger.error(f"Error reading from PostgreSQL table {self.table}: {e}")
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred while reading from {self.table}: {e}")
                return None

    def bulk_write(self, items: List[dict]):
        """
        Insert multiple items using batch execution.
        """
        if not self.conn or not items:
            return False
        columns = ", ".join(items[0].keys())
        values = ", ".join([f"%({k})s" for k in items[0].keys()])
        query = f"INSERT INTO {self.table} ({columns}) VALUES ({values}) ON CONFLICT DO NOTHING"

        try:
            with self.conn.cursor() as cur:
                cur.executemany(query, items)
                self.conn.commit()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error writing multiple records to PostgreSQL: {e}")
            return False

    def update(self, key: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """
        Update existing record(s) based on the key.
        """
        if not self.conn:
            return False
        key_column, key_value = next(iter(key.items()))
        set_clause = ", ".join([f"{k} = %s" for k in data.keys()])
        query = f"UPDATE {self.table} SET {set_clause} WHERE {key_column} = %s"

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, list(data.values()) + [key_value])
                self.conn.commit()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error updating PostgreSQL: {e}")
            return False

    def close(self):
        """ Close the database connection. """
        if self.conn:
            self.conn.close()
            
    def delete(self, key: Dict[str, Any]) -> bool:
        """
        Delete existing record(s) based on the key_conditions.

        """
        if not self.conn or not key:
            logger.error("Delete: No connection or no key conditions specified.")
            return False

        where_clauses = []
        values = []
        for col, val in key.items():
            where_clauses.append(f'"{col}" = %s')
            values.append(val)
        
        where_expression = " AND ".join(where_clauses)
        query = f"DELETE FROM {self.table} WHERE {where_expression}"

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, tuple(values))
                if cur.rowcount == 0:
                    logger.info(f"Warning: Delete affected 0 rows for key conditions {key}. Item(s) might not exist.")
                else:
                    logger.info(f"Successfully deleted {cur.rowcount} row(s) matching {key}.")
                self.conn.commit()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error deleting from PostgreSQL table {self.table}: {e}")
            self.conn.rollback()
            return False