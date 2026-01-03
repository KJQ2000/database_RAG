"""
main.py

Pipeline for processing user queries in a jewelry store context. 

Features:
- Intent classification
- Knowledge retrieval (vector DB)
- SQL generation & execution
- Answer generation (RAG)
- Supports 'database_query' and 'store_info' intents
"""

import os
import argparse
from dotenv import load_dotenv
from jinja2 import Template
from flask import jsonify

from src.shared.logger import Logger
from src.shared.utils import load_config
from src.shared.constant import *
from src.shared.database import Database
from src.pipeline.intent_classifier_pipeline import IntentClassifier
from src.pipeline.build_vector_db_pipeline import BuildVectorDB
from src.pipeline.answer_generator_pipeline import AnswerGenerator
from src.pipeline.sql_generator_pipeline import SQLAgent

# Load environment variables
load_dotenv()

# Initialize logger
logger = Logger().get_logger()


def run_intent_classifier(query: str, configs: dict) -> str:
    """
    Classify the user intent.

    Args:
        query (str): The user question.
        configs (dict): Loaded configuration dictionary.

    Returns:
        str: Classified intent ('database_query', 'store_info', 'unclear', etc.)
    """
    try:
        intent_system_prompt = configs['intent_system_prompt']
        classifier = IntentClassifier(intent_system_prompt=intent_system_prompt)
        intent = classifier.classify_intent(query)
        logger.info(f"Intent classified as: {intent}")
        return intent
    except Exception as e:
        logger.error("Failed to classify intent.", exc_info=True)
        return 'unclear'


def build_vector_db(query: str, intent: str, configs: dict):
    """
    Build or retrieve vector database for given intent.

    Args:
        query (str): User question
        intent (str): User intent
        configs (dict): Loaded configuration

    Returns:
        BuildVectorDB: Instance with retrieval capability
    """
    try:
        persist_path = os.path.join(os.getcwd(), VECTOR_DB_DIR)
        hash_store_filename = DB_RAG_HASH_STORE_FILENAME if intent == 'database_query' else SI_RAG_HASH_STORE_FILENAME
        knowledge_file = DB_RAG_KNOWLEDGE_BASE_PATH if intent == 'database_query' else SI_RAG_KNOWLEDGE_BASE_PATH

        vb = BuildVectorDB(
            persist_path=persist_path,
            embedding_model="text-embedding-3-small",
            hash_store_filename=hash_store_filename,
            intent=intent
        )
        vb.build_from_knowledge_base_if_changed(knowledge_file=knowledge_file)
        return vb
    except Exception as e:
        logger.error("Vector DB build/retrieve failed.", exc_info=True)
        raise RuntimeError("Failed to build or load vector DB") from e


def handle_store_info(query: str, vb: BuildVectorDB, configs: dict) -> str:
    """
    Handle 'store_info' intent: retrieve relevant documents and generate answer.

    Args:
        query (str): User question
        vb (BuildVectorDB): Vector DB instance
        configs (dict): Configuration dict

    Returns:
        str: Generated answer
    """
    try:
        chunks = vb.retrieve(query=query, k=8)
        system_prompt = configs['si_rag_answer_generator_system_prompt']

        answer_generator = AnswerGenerator(
            system_prompt=system_prompt,
            intent='store_info'
        )
        answer = answer_generator.generate_answer(
            query=query,
            retrieved_documents=chunks
        )
        return answer
    except Exception as e:
        logger.error("Failed to handle store_info intent.", exc_info=True)
        return "Sorry, I couldn’t generate an answer for this question."


def handle_database_query(query: str, vb: BuildVectorDB, configs: dict) -> str:
    """
    Handle 'database_query' intent: generate SQL, execute, and summarize results.

    Args:
        query (str): User question
        vb (BuildVectorDB): Vector DB instance
        configs (dict): Configuration dict

    Returns:
        str: Generated answer based on SQL result
    """
    try:
        # Initialize database connection
        db = Database(
            host=os.environ["HOST"],
            port=os.environ["PORT"],
            database=os.environ["DATABASE"],
            user=os.environ["USER"],
            password=os.environ["PASSWORD"]
        )

        # Retrieve context for SQL generation
        system_prompt_template = configs["db_rag_sql_generator"]
        schema = os.getenv("SCHEMA")
        chunks = vb.retrieve(query=query, k=8)

        dictionary_text = "\n".join(
            c.page_content if hasattr(c, "page_content") else str(c)
            for c in chunks
        )

        final_system_prompt = Template(system_prompt_template).render(
            schema=schema,
            dictionary=dictionary_text
        )

        # Generate SQL
        sql_agent = SQLAgent(
            system_prompt=final_system_prompt,
            model=DB_RAG_LLM,
            temperature=DB_RAG_TEMPERATURE,
            max_tokens=DB_RAG_MAX_TOKEN
        )
        sql_query = sql_agent.generate_sql(user_question=query)
        logger.info(f"Generated SQL: {sql_query}")

        # Execute SQL
        df = db.select_raw(sql_query)
        if df is None or df.empty:
            logger.warning("SQL execution returned no results.")
            return "No results found or query error."

        data_str = df.to_csv(index=False)

        # Prepare context for answer generator
        context = f"SQL QUERY:\n{sql_query}\n\nRESULT:\n{data_str}"

        system_prompt = configs['db_rag_answer_generator_system_prompt']
        answer_generator = AnswerGenerator(
            system_prompt=system_prompt,
            intent='database_query'
        )

        answer = answer_generator.generate_answer(
            query=query,
            retrieved_documents=context,
            mode='db_summarize'
        )

        return answer

    except Exception as e:
        logger.error("Failed to handle database_query intent.", exc_info=True)
        return "Sorry, I couldn’t generate an answer for this question."


def main(query: str, run_intent: bool = True, run_answer: bool = True):
    """
    Main pipeline function.

    Args:
        query (str): User question
        run_intent (bool): Whether to run intent classification
        run_answer (bool): Whether to run answer generation

    Returns:
        str: Final answer
    """
    try:
        # Load config
        config_file_path = os.path.join(os.getcwd(), 'config', 'config.yaml')
        configs = load_config(config_file_path)

        intent = None

        if run_intent:
            intent = run_intent_classifier(query, configs)

        if run_answer:
            if not intent:
                intent = run_intent_classifier(query, configs)

            vb = build_vector_db(query, intent, configs)

            if intent == 'store_info':
                return handle_store_info(query, vb, configs)

            elif intent == 'database_query':
                return handle_database_query(query, vb, configs)

            elif intent == 'unclear':
                logger.warning("User query could not be classified.")
                return "Sorry, I’m not sure what you are asking."

            else:
                logger.warning(f"Unknown intent: {intent}")
                return "Unknown intent."

        return "Pipeline executed with no output."

    except Exception as e:
        logger.error("Pipeline execution failed.", exc_info=True)
        return "An unexpected error occurred during query processing."
