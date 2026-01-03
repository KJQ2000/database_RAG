import os
from dotenv import load_dotenv
from typing import List, Union

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, Document

from src.shared.logger import Logger
from src.shared.constant import (
    DB_RAG_LLM,
    DB_RAG_TEMPERATURE,
    DB_RAG_MAX_TOKEN,
    SI_RAG_LLM,
    SI_RAG_TEMPERATURE,
    SI_RAG_MAX_TOKEN
)

# Load environment variables
load_dotenv()


class AnswerGenerator:
    """
    AnswerGenerator is responsible for generating final user-facing answers
    using retrieved documents and a domain-specific system prompt.

    This class DOES NOT perform retrieval.
    It strictly performs answer synthesis (RAG generation).

    Supported intents:
        - database_query
        - store_info
    """

    def __init__(self, system_prompt: str, intent: str):
        """
        Initializes the AnswerGenerator.

        Args:
            system_prompt (str): System prompt defining answer behavior.
            intent (str): Either 'database_query' or 'store_info'.

        Raises:
            ValueError: If invalid arguments are provided.
            EnvironmentError: If OpenAI API key is missing.
            RuntimeError: If LLM initialization fails.
        """
        self.logger = Logger().get_logger()
        self.logger.info("Initializing AnswerGenerator...")

        # Validate intent
        if intent not in ("database_query", "store_info"):
            self.logger.error(f"Invalid intent provided: {intent}")
            raise ValueError("intent must be 'database_query' or 'store_info'")

        self.intent = intent

        # Validate system prompt
        if not system_prompt or not isinstance(system_prompt, str):
            self.logger.error("Invalid system_prompt provided.")
            raise ValueError("system_prompt must be a non-empty string")

        self.system_prompt = system_prompt

        # Load API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.error("OPENAI_API_KEY not found.")
            raise EnvironmentError("OPENAI_API_KEY is required")

        # Select model configuration
        if self.intent == "database_query":
            self.model = DB_RAG_LLM
            self.temperature = DB_RAG_TEMPERATURE
            self.max_tokens = DB_RAG_MAX_TOKEN
        else:
            self.model = SI_RAG_LLM
            self.temperature = SI_RAG_TEMPERATURE
            self.max_tokens = SI_RAG_MAX_TOKEN

        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.api_key
            )
            self.logger.info(
                f"LLM initialized: model={self.model}, "
                f"temperature={self.temperature}, max_tokens={self.max_tokens}"
            )
        except Exception as e:
            self.logger.error("Failed to initialize LLM", exc_info=True)
            raise RuntimeError("LLM initialization failed") from e

    def generate_answer(
        self,
        query: str,
        retrieved_documents: Union[List[Document], str, None] = None,
        mode:str =None
    ) -> str:
        """
        Generates a final answer for the user using retrieved documents.

        Args:
            user_question (str): The user's question.
            retrieved_documents (List[Document]): Documents returned by vector search.

        Returns:
            str: Final answer safe for direct user display.

        Behavior:
            - Uses ONLY retrieved context
            - Does NOT hallucinate missing information
            - Returns a fallback message if context is empty
        """
        self.logger.info("Generating answer...")

        if not query or not isinstance(query, str):
            self.logger.warning("Invalid query.")
            return "Sorry, I couldn't understand your question."

        if not retrieved_documents:
            self.logger.warning("No retrieved documents provided.")
            return (
                "I’m sorry, I don’t have enough information to answer this question "
                "based on the available knowledge."
            )

        try:
            if mode == 'db_summarize':
                context = retrieved_documents
            else:
                # Combine retrieved documents into context
                context = "\n\n".join(
                    f"[Source {i+1}]\n{doc.page_content}"
                    for i, doc in enumerate(retrieved_documents)
                )

            # Construct user prompt
            user_prompt = f"""
                Use ONLY the information below to answer the question.

                Context:
                {context}

                Question:
                {query}

                If the answer cannot be found in the context, say:
                "I don’t have enough information to answer this question."
                """

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt.strip())
            ]
            
            self.logger.debug(messages)

            response = self.llm(messages)

            answer = response.content.strip()

            self.logger.info("Answer generated successfully.")
            return answer

        except Exception as e:
            self.logger.error("Answer generation failed.", exc_info=True)
            return (
                "Sorry, something went wrong while generating the answer. "
                "Please try again later."
            )
