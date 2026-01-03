import os

DATABASE_URL = 'postgresql://JunQiang:UBjUWi4UNOlyiMQy22_ZsQ@konghin-imtool-7458.6xw.aws-ap-southeast-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full'
SCHEMA = 'konghin'


USER_SEQ = 'usr_id_seq'
BOOKING_SEQ = 'book_id_seq'
CUSTOMER_SEQ = 'cust_id_seq'
STOCK_SEQ = 'stk_id_seq'
SALE_SEQ = 'sale_id_seq'
SALESMAN_SEQ = 'slm_id_seq'
PURCHASE_SEQ = 'pur_id_seq'
BOOK_PAYMENT_SEQ = 'bp_id_seq'
CPAT_SEQ = 'cpat_id_seq'


CURRENT_DIR = os.path.abspath(os.getcwd())
IMPORT_DIR = os.path.join(os.path.abspath(os.getcwd()), 'system_files','Import')
SUCCESS_IMPORT_DIR = os.path.join(os.path.abspath(os.getcwd()), 'system_files','Import','ARCHIVED','SUCCESS')
FAILED_IMPORT_DIR = os.path.join(os.path.abspath(os.getcwd()), 'system_files','Import','ARCHIVED','FAILED')
IMG_STORE_DIR = os.path.join(os.path.abspath(os.getcwd()), 'website','static','pattern')
BATCH_INSERT_LIMIT = 1000

PY_IMPORT_FILE = os.path.join(os.path.abspath(os.getcwd()), 'website','Import.py')

LOG_DIR = os.path.join(os.path.abspath(os.getcwd()), 'system_files','Log')

APP_SECRET_KEY = b'k0ngh1n888'
