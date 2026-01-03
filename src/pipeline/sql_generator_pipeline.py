import os
from dotenv import load_dotenv
from typing import Optional

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from src.shared.logger import Logger

# Load environment variables
load_dotenv()


class SQLAgent:
    """
    SQLAgent generates PostgreSQL SELECT queries
    from natural language questions.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ):
        self.logger = Logger().get_logger()
        self.logger.info("Initializing SQLAgent...")

        if not system_prompt or not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a non-empty string")

        self.system_prompt = system_prompt

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY is required but not set")

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=False,
            openai_api_key=self.api_key
        )

        self.prompt = self._build_prompt_template()

        self.logger.info(
            f"SQLAgent initialized with model={model}, "
            f"temperature={temperature}, max_tokens={max_tokens}"
        )

    def _build_prompt_template(self) -> ChatPromptTemplate:
        system_msg = SystemMessagePromptTemplate.from_template(self.system_prompt,template_format="jinja2")
        human_msg = HumanMessagePromptTemplate.from_template("{{ user_question }}", template_format="jinja2")

        return ChatPromptTemplate.from_messages([system_msg, human_msg])

    def generate_sql(self, user_question: str) -> str:
        """
        Generates SQL from user question.
        """

        if not user_question or not isinstance(user_question, str):
            self.logger.warning("Invalid user_question provided.")
            return "I am not sure on this question."

        self.logger.info(f"Generating SQL for question: {user_question}")

        try:
            messages = self.prompt.format_prompt(
                user_question=user_question
            ).to_messages()

            response = self.llm(messages)
            sql = response.content.strip()

            self.logger.debug(f"Generated SQL: {sql}")
            return sql

        except Exception:
            self.logger.error("SQL generation failed.", exc_info=True)
            return "I am not sure on this question."

