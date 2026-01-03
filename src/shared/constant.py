SCHEMA = 'konghin'

INTENTS = [
    "database_query",
    "store_info",
    "unclear"
]

INTENT_LLM = "gpt-4o-mini"
INTENT_TEMPERATURE = 0
INTENT_MAX_TOKEN = 10

VECTOR_DB_DIR = 'vector_db'


DB_RAG_VECTOR_DB_FILENAME = 'db_rag_index.pkl'
DB_RAG_HASH_STORE_FILENAME = 'db_rag_hashes.json'
DB_RAG_VECTOR_DB_PATH = f'{VECTOR_DB_DIR}/{DB_RAG_VECTOR_DB_FILENAME}'
DB_RAG_KNOWLEDGE_BASE_PATH = r'C:\Users\keong\OneDrive\E-commerce\Manual Guide\Bulk Insert Template\Bilingual README.txt'

DB_RAG_LLM = "gpt-4o-mini"
DB_RAG_TEMPERATURE = 0.2
DB_RAG_MAX_TOKEN = 512

SI_RAG_VECTOR_DB_FILENAME = 'store_info_index.pkl'
SI_RAG_HASH_STORE_FILENAME = 'store_info_hashes.json'
SI_RAG_VECTOR_DB_PATH = f'{VECTOR_DB_DIR}/{DB_RAG_VECTOR_DB_FILENAME}'
SI_RAG_KNOWLEDGE_BASE_PATH = r'C:\OpenAI\RnD\database_RAG\knowledge\knowledge.txt'

SI_RAG_LLM = "gpt-4o-mini"
SI_RAG_TEMPERATURE = 0.3
SI_RAG_MAX_TOKEN = 512