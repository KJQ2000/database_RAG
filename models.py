import os
import psycopg2
from psycopg2 import sql
import logging
import dictionary as dic
import numpy as np
from datetime import datetime
import pandas as pd
import json
from barcode import BarcodeGenerator


SEQUENCES = {
    'users': dic.USER_SEQ,
    'booking': dic.BOOKING_SEQ,
    'customer': dic.CUSTOMER_SEQ,
    'stock': dic.STOCK_SEQ,
    'sale': dic.SALE_SEQ,
    'salesman': dic.SALESMAN_SEQ,
    'purchase': dic.PURCHASE_SEQ,
    'book_payment':dic.BOOK_PAYMENT_SEQ,
    'category_pattern_mapping':dic.CPAT_SEQ
}

PREFIX = {
    'users': 'USR',
    'booking': 'BOOK',
    'customer': 'CUST',
    'stock': 'STK',
    'sale': 'SALE',
    'salesman': 'SLM',
    'purchase': 'PUR',
    'book_payment':'BP',
    'category_pattern_mapping':'CPAT'
}


log_file = dic.LOG_DIR+str(datetime.now().strftime("%Y_%m_%d"))+'.log'
logging.basicConfig(filename=log_file,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Database: 
    def __init__(self, database_url: str):
        self.conn = psycopg2.connect(database_url)
        self.cursor = self.conn.cursor()
        self.schema = 'konghin'
        
    def select_raw(self, query: str, params: tuple = None, js: bool = False):
        """
        Executes a raw SQL query with optional parameters.

        Args:
            query (str): The SQL query to execute.
            params (tuple, optional): Parameters to safely substitute into the query.
            js (bool, optional): If True, returns the result as a JSON string.

        Returns:
            pd.DataFrame or str: DataFrame of results or JSON if js=True.
        """
        try:
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
        
        
    
    def select(self, table: str, columns: list = None, where: str = None, js: bool = False):
        """
        Selects data from a specified table.

        Args:
            table (str): The table name from which to select data.
            columns (list, optional): List of column names to select. If None, selects all columns. Defaults to None.
            where (str, optional): SQL condition for filtering rows. Defaults to None.
            json (bool, optional): If True, returns the result as a JSON string. If False, returns as a DataFrame. Defaults to False.

        Returns:
            pd.DataFrame or str: A DataFrame of the selected data or a JSON string if json is True. Returns None if an error occurs.
        """
        if columns:
            query = sql.SQL("SELECT {columns} FROM {schema}.{table}").format(
                columns=sql.SQL(', ').join(map(sql.Identifier, columns)),
                schema=sql.Identifier(self.schema),
                table=sql.Identifier(table)
            )
        else:
            query = sql.SQL("SELECT * FROM {schema}.{table}").format(
                schema=sql.Identifier(self.schema),
                table=sql.Identifier(table)
            )

        if where:
            query += sql.SQL(" WHERE {where}").format(where=sql.SQL(where))

        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            df = pd.DataFrame(np.array(results))
            colnames = [desc[0] for desc in self.cursor.description]
            df.columns = colnames
            logging.info(f"Successfully selected data from {self.schema}.{table}.")
            logging.info(f"Query: {query.as_string(self.conn)}")
            if json:
                return json.loads(df.to_json(orient='records', date_format='iso'))
            else:
                return df
        except psycopg2.Error as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Database error: {e.pgcode} - {e.pgerror}")
            logging.error(f"Error details: {e.diag.message_detail}")
            return None
        except Exception as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Unexpected error: {e}")
            return None

    def insert(self, table: str, values: list, columns: list = None) -> None:
        base_query = sql.SQL("INSERT INTO {schema}.{table}").format(
            schema=sql.Identifier(self.schema),
            table=sql.Identifier(table)
        )

        sequence_name = SEQUENCES.get(table.lower())

        if sequence_name ==None:
            logging.error(f"Table not found. Please ensure u entered correct table name.")
            self.conn.rollback()
            return None
        else:
            seq = self.get_nextval(sequence_name)

        pk = str(PREFIX.get(table))+'_'+str(seq)
        
        if table.lower() == 'stock':
            gp = values[columns.index('stk_gold_cost')]
            labor = values[columns.index('stk_labor_cost')]
            #generate stk_barcode
            barcode_gen = BarcodeGenerator()
            unique_key = barcode_gen.generate(gp,labor,str(seq))
            columns = ['stk_barcode'] + columns
            values = [unique_key] + values

        if columns:
            columns = [item1 for item1, item2 in zip(columns, values) if item2 != '']
            values = [item2 for item2 in values if item2 != '']
            query = base_query + sql.SQL(" ({id_col}, {columns}) VALUES ({id}, {values})").format(
                id_col = sql.Identifier(str(PREFIX.get(table)).lower()+'_id'),
                columns=sql.SQL(', ').join(map(sql.Identifier, columns)),
                id = sql.Placeholder(),
                values=sql.SQL(', ').join(sql.Placeholder() * len(values))
            )
        else:
            values = [item2 for item2 in values if item2 != '']
            query = base_query + sql.SQL(" VALUES ({id}, {values})").format(
                id = sql.Placeholder(),
                values=sql.SQL(', ').join(sql.Placeholder() * len(values))
            )
        
        values = [pk] + values

        
        try:
            self.cursor.execute(query, values)
            self.conn.commit()
            logging.info(f"Successfully inserted data into {self.schema}.{table}.")
            logging.info(f"Query: {query.as_string(self.conn)}")
            logging.info(f"Values: {values}")
        except (psycopg2.Error, psycopg2.DatabaseError) as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Database error: {e.pgcode} - {e.pgerror}")
            logging.error(f"Error details: {e.diag.message_detail}")
            self.conn.rollback()
        except Exception as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Unexpected error: {e}")
            self.conn.rollback()



    def update(self, table: str, set_columns: list, set_values: list, where: str) -> None:
        # Filter columns and values, allowing explicit NULLs
        clean_columns, clean_values = [], []
        for col, val in zip(set_columns, set_values):
            if val not in ('None', ''):
                clean_columns.append(col)
                clean_values.append(val)
            else:
                clean_columns.append(col)
                clean_values.append(None)  # Explicitly set value to None (for NULL in SQL)
        
        if table != 'metadata':
            last_update = str(PREFIX.get(table)).lower() + '_last_update'
            clean_columns.append(last_update)
            clean_values.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Construct the SET clause with proper handling for NULL
        set_clause = sql.SQL(', ').join(
            sql.SQL("{col} = {val}").format(
                col=sql.Identifier(col),
                val=sql.SQL('NULL') if value is None else sql.Placeholder()
            )
            for col, value in zip(clean_columns, clean_values)
        )
        
        query = sql.SQL("UPDATE {schema}.{table} SET {set_clause} WHERE {where}").format(
            schema=sql.Identifier(self.schema),
            table=sql.Identifier(table),
            set_clause=set_clause,
            where=sql.SQL(where)
        )
        
        try:
            # Filter out None values from clean_values, as they are not needed for placeholders
            clean_values_for_execute = [val for val in clean_values if val is not None]
            self.cursor.execute(query, clean_values_for_execute)
            self.conn.commit()
            logging.info(f"Successfully updated data in {self.schema}.{table}.")
            logging.info(f"Query: {query.as_string(self.conn)}")
            logging.info(f"Values: {clean_values_for_execute}")
        except psycopg2.Error as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Database error: {e.pgcode} - {e.pgerror}")
            logging.error(f"Error details: {e.diag.message_detail}")
            self.conn.rollback()
        except Exception as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Unexpected error: {e}")
            self.conn.rollback()


    def delete(self, table: str, where: str) -> None:
        query = sql.SQL("DELETE FROM {schema}.{table} WHERE {where}").format(
            schema=sql.Identifier(self.schema),
            table=sql.Identifier(table),
            where=sql.SQL(where)
        )

        try:
            self.cursor.execute(query)
            self.conn.commit() 
            logging.info(f"Successfully deleted data from {self.schema}.{table}.")
            logging.info(f"Query: {query.as_string(self.conn)}")
        except psycopg2.Error as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Database error: {e.pgcode} - {e.pgerror}")
            logging.error(f"Error details: {e.diag.message_detail}")
            self.conn.rollback()
        except Exception as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Unexpected error: {e}")
            self.conn.rollback()

    def get_nextval(self, sequence_name: str):
        query = sql.SQL("select nextval('{schema}.{seq}')").format(
            schema=sql.Identifier(dic.SCHEMA),
            seq=sql.Identifier(sequence_name)
        )

        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()[0]
            logging.info(f"Next value of sequence {sequence_name}: {result}")
            return result
        except psycopg2.Error as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Database error: {e.pgcode} - {e.pgerror}")
            logging.error(f"Error details: {e.diag.message_detail}")
            return None
        except Exception as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Unexpected error: {e}")
            return None
    
    def get_currval(self, sequence_name: str):
        query = sql.SQL("select currval('{schema}.{seq}')").format(
            schema=sql.Identifier(dic.SCHEMA),
            seq=sql.Identifier(sequence_name)
        )

        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()[0]
            logging.info(f"current value of sequence {sequence_name}: {result}")
            return result
        except psycopg2.Error as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Database error: {e.pgcode} - {e.pgerror}")
            logging.error(f"Error details: {e.diag.message_detail}")
            return None
        except Exception as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Unexpected error: {e}")
            return None
    
    def drop_seq(self, sequence_name:str):
        query = sql.SQL("DROP SEQUENCE {schema}.{seq}").format(
            schema=sql.Identifier(dic.SCHEMA),
            seq=sql.Identifier(sequence_name)
        )

        try:
            self.cursor.execute(query)
            logging.info(f"sequence {sequence_name} dropped!")
            return True
        except psycopg2.Error as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Database error: {e.pgcode} - {e.pgerror}")
            logging.error(f"Error details: {e.diag.message_detail}")
            return None
        except Exception as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Unexpected error: {e}")
            return None
        
    def create_stk_seq(self, sequence_name:str):
        query = sql.SQL("CREATE SEQUENCE {schema}.{seq} INCREMENT BY 1 MINVALUE 1 MAXVALUE 999999 START 100001 NO CYCLE;").format(
            schema=sql.Identifier(dic.SCHEMA),
            seq=sql.Identifier(sequence_name)
        )

        try:
            self.cursor.execute(query)
            logging.info(f"sequence {sequence_name} created!")
            return True
        except psycopg2.Error as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Database error: {e.pgcode} - {e.pgerror}")
            logging.error(f"Error details: {e.diag.message_detail}")
            return None
        except Exception as e:
            logging.error(f"Query: {query.as_string(self.conn)}")
            logging.error(f"Unexpected error: {e}")
            return None
    

    def batch_insert(self, table: str, values: list, columns: list = None) -> None:
        base_query = "INSERT INTO {schema}.{table}".format(
            schema=self.schema,
            table=table
        )
        
        if table.lower() == 'stock' and columns:
            columns = ['stk_barcode'] + columns

        if columns:
            query = base_query + " ({id_col}, {columns}) VALUES ".format(
                id_col = str(PREFIX.get(table)).lower()+'_id',
                columns=', '.join(columns)
            )
        else:
            query = base_query + " VALUES "
            
        values_statement = ''
        
        # print('until_values: ',query)

        for row in range(len(values)):
            seq = self.get_nextval(SEQUENCES.get(table))
            
            if table.lower() == 'stock':
                gp = values[row][columns.index('stk_gold_cost')-1]
                labor = values[row][columns.index('stk_labor_cost')-1]
                #generate stk_barcode
                barcode_gen = BarcodeGenerator()
                barcode = barcode_gen.generate(gp,labor,str(seq))
                values[row] = [barcode] + values[row]

            pk = str(PREFIX.get(table))+'_'+str(seq)
            formatted_string = ', '.join(f"'{item}'" for item in values[row])
            values_statement += "('" + pk +"',"+ formatted_string+"),"

        values_statement = values_statement[:-1]
        
        insert_query = (query+values_statement).replace("'nan'",'null')

        # print('Full query: ',insert_query)


        try:
            self.cursor.execute(insert_query)
            self.conn.commit()
            logging.info(f"Successfully inserted data into {self.schema}.{table}.")
            logging.info(f"Query: {insert_query}")
        except (psycopg2.Error, psycopg2.DatabaseError) as e:
            logging.error(f"Query: {insert_query}")
            logging.error(f"Database error: {e.pgcode} - {e.pgerror}")
            logging.error(f"Error details: {e.diag.message_detail}")
            self.conn.rollback()
        except Exception as e:
            logging.error(f"Query: {insert_query}")
            logging.error(f"Unexpected error: {e}")
            self.conn.rollback()
    
    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    

# if __name__ == '__main__':
#     db = Database(os.environ["DATABASE_URL"])
    
    # Example usage
    # db.insert(table='abcd', values=['kjunqiang@gmail.com', '11115354', 'JunQiang'])
    # db.insert(table='users', values=['yckng00@gmail.com', '00121800', 'YinChew'])
    # db.update('users', set_columns=['username'], set_values=['qwer'], where='id=3')
    # stock = db.select('stock')
    # print(pd.DataFrame(stock))
    # db.select('users', columns=['username'], where='id=3')
    # db.delete('users', where='id=3')
    # db.batch_insert(table='users', values=[['abcd@gmail.com', '1234567', 'abcd'], ['efgh@gmail.com', 'abcdefg', 'efgh'], ['ijkl@gmail.com', 'abcd1234', 'abcdefgh1234']])

# import psycopg2
# from psycopg2 import sql
# import os, re

# class Database():

#     # def __init__(self, conn):
#     #     self.conn = conn
#     #     self.cursor = conn.cursor()
#     #     self.username = ""

#     def __init__(self, database_url:str):
#         self.conn = psycopg2.connect(database_url)
#         self.cursor = self.conn.cursor()
#         self.schema = 'konghin'

#         self.username = ""
        

    # def authenticate(self, email, password):

    #     with self.cursor as cur:
    #         query = sql.SQL("SELECT * FROM konghin.users WHERE usr_email = %s AND usr_password = %s;")
    #         cur.execute(query, (email,password))
    #         res = cur.fetchone()
    #         self.conn.commit()
            
    #         if res:
    #             self.username = res[3]
    #             return True
    #         else:
    #             return False
            
    # def get_username(self):
    #     return self.username
    

    # def register_new_user(self, username, email, password):
    #     # Check if account exists using MySQL
    #     with self.cursor as cur:
    #         query = sql.SQL("SELECT * FROM konghin.users WHERE usr_email = %s;")
    #         cur.execute(query, (email))
    #         res = cur.fetchone()

    #         # If account exists show error and validation checks
    #         if res:
    #             msg = 'Account already exists!'
    #         elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
    #             msg = 'Invalid email address!'
    #         elif not re.match(r'[A-Za-z0-9]+', username):
    #             msg = 'Username must contain only characters and numbers!'
    #         elif not username or not password or not email:
    #             msg = 'Please fill out the form!'
    #         else:
    #             # Account doesn't exist, and the form data is valid, so insert the new account into the accounts table
    #             cur.execute('INSERT INTO konghin.users VALUES (NULL, %s, %s, %s)', (email, password, username,))
    #             self.conn.commit()
    #             msg = 'You have successfully registered!'