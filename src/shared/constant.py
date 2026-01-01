KNOWLEDGE_BASE_PATH = r'C:\Users\keong\OneDrive\E-commerce\Manual Guide\Bulk Insert Template\Bilingual README.txt'

SCHEMA = 'konghin'

INTENTS = [
    "database_query",
    "store_info",
    "unclear"
]

INTENT_LLM = "gpt-4o-mini"
INTENT_TEMPERATURE = 0
INTENT_MAX_TOKEN = 10

DB_RAG_VECTOR_DB_DIR = 'vector_db'
DB_RAG_VECTOR_DB_FILENAME = 'index.pkl'
DB_RAG_VECTOR_DB_PATH = f'vector_db/{DB_RAG_VECTOR_DB_FILENAME}'

DB_RAG_LLM = "gpt-4o-mini"
DB_RAG_TEMPERATURE = 0.2
DB_RAG_MAX_TOKEN = 512