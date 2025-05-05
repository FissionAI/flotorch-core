import psycopg2
from psycopg2.extras import RealDictCursor
from flotorch_core.storage.db.db_storage import DBStorage
from typing import List, Dict, Any

class PostgresDB(DBStorage):
    def __init__(self, dbname: str, user: str, password: str, host: str = "localhost", port: int = 5432):
        self.dbname = dbname
        self.user = user
        self.password = password
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
            print(f"Error connecting to PostgreSQL: {e}")
            return None

    def write(self, item: dict, table: str):
        """
        Insert a single item into the specified table.
        """
        if not self.conn:
            return False
        columns = ", ".join(item.keys())
        values = ", ".join([f"%({k})s" for k in item.keys()])
        query = f"INSERT INTO {table} ({columns}) VALUES ({values}) ON CONFLICT DO NOTHING"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, item)
                self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Error writing to PostgreSQL: {e}")
            return False

    def read(self, key: Dict[str, Any], table: str) -> dict:
        """
        Retrieve a single item based on key.
        """
        if not self.conn:
            return None
        key_column, key_value = next(iter(key.items()))
        query = f"SELECT * FROM {table} WHERE {key_column} = %s"

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (key_value,))
                return cur.fetchone()
        except psycopg2.Error as e:
            print(f"Error reading from PostgreSQL: {e}")
            return None

    def bulk_write(self, items: List[dict], table: str):
        """
        Insert multiple items using batch execution.
        """
        if not self.conn or not items:
            return False
        columns = ", ".join(items[0].keys())
        values = ", ".join([f"%({k})s" for k in items[0].keys()])
        query = f"INSERT INTO {table} ({columns}) VALUES ({values}) ON CONFLICT DO NOTHING"

        try:
            with self.conn.cursor() as cur:
                cur.executemany(query, items)
                self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Error writing multiple records to PostgreSQL: {e}")
            return False

    def update(self, key: Dict[str, Any], data: Dict[str, Any], table: str) -> bool:
        """
        Update existing record(s) based on the key.
        """
        if not self.conn:
            return False
        key_column, key_value = next(iter(key.items()))
        set_clause = ", ".join([f"{k} = %s" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {key_column} = %s"

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, list(data.values()) + [key_value])
                self.conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Error updating PostgreSQL: {e}")
            return False

    def close(self):
        """ Close the database connection. """
        if self.conn:
            self.conn.close()
