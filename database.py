import psycopg2
import pandas as pd
import logging

log_file = "db_query.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Database:
    def __init__(self, host: str, port: str, database: str, user: str, password: str):
        self.conn = psycopg2.connect(host=host, port=port, database=database, user=user, password=password)
        self.cursor = None
        self.schema = 'konghin'
        self.conn.autocommit = False

    def select_raw(self, query: str, params: tuple = None, js: bool = False):
        try:
            if self.cursor is None or self.cursor.closed:
                self.cursor = self.conn.cursor()

            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            colnames = [desc[0] for desc in self.cursor.description]

            df = pd.DataFrame(results, columns=colnames)
            logging.info(f"Successfully executed query: {query}")

            if js:
                return df.to_json(orient='records', date_format='iso')
            return df
        except psycopg2.Error as e:
            logging.error(f"Database error: {e.pgcode} - {e.pgerror}")
            return None
        except Exception as e:
            logging.error(f"Query failed: {e}")
            return None

    def __del__(self):
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
