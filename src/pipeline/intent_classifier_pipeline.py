import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from src.shared.logger import Logger
from src.shared.constant import INTENT_LLM, INTENT_TEMPERATURE, INTENT_MAX_TOKEN

# Load environment variables from .env
load_dotenv()

class IntentClassifier:
    """
    IntentClassifier is responsible for classifying user questions
    into predefined intents using an OpenAI LLM model.

    Attributes:
        intent_system_prompt (str): The system prompt defining the classification rules.
        api_key (str): OpenAI API key loaded from environment variables.
        model (str): The LLM model name.
        temperature (float): Sampling temperature for model outputs.
        max_tokens (int): Maximum tokens the model can return.
        intent_llm (ChatOpenAI): Initialized LangChain ChatOpenAI LLM instance.
        logger (logging.Logger): Centralized logger instance.

    Methods:
        classify_intent(user_question: str) -> str:
            Classifies a user's question into a single intent.
    """

    def __init__(self, intent_system_prompt: str):
        """
        Initializes the IntentClassifier with the required configurations.

        Args:
            intent_system_prompt (str): The system prompt for intent classification.

        Raises:
            ValueError: If the system prompt is None or empty.
            RuntimeError: If the LLM initialization fails.
        """
        # Initialize logger
        self.logger = Logger().get_logger()
        self.logger.info("Initializing IntentClassifier...")

        # Validate system prompt
        if not intent_system_prompt or not isinstance(intent_system_prompt, str):
            self.logger.error("Invalid 'intent_system_prompt' provided.")
            raise ValueError("'intent_system_prompt' must be a non-empty string.")

        self.intent_system_prompt = intent_system_prompt

        # Load API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.error("OPENAI_API_KEY not found in environment variables.")
            raise EnvironmentError("OPENAI_API_KEY is required but not set.")

        # Load model configurations from constants
        self.model = INTENT_LLM
        self.temperature = INTENT_TEMPERATURE
        self.max_tokens = INTENT_MAX_TOKEN

        # Initialize LLM
        try:
            self.intent_llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.api_key
            )
            self.logger.info(
                f"LLM initialized successfully: model={self.model}, "
                f"temperature={self.temperature}, max_tokens={self.max_tokens}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatOpenAI LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}") from e

    def classify_intent(self, user_question: str) -> str:
        """
        Classifies the user's question into one of the predefined intents.

        Args:
            user_question (str): The question input by the user.

        Returns:
            str: The classified intent in lowercase. Returns 'error' if classification fails.

        Logs:
            INFO: When a question is being classified and when the result is obtained.
            ERROR: If an exception occurs during classification.
        """
        if not user_question or not isinstance(user_question, str):
            self.logger.warning("Empty or invalid user_question provided.")
            return "error"

        self.logger.info(f"Classifying user question: '{user_question}'")

        try:
            # Construct messages for the LLM
            messages = [
                SystemMessage(content=self.intent_system_prompt),
                HumanMessage(content=user_question)
            ]

            # Get response from LLM
            response = self.intent_llm(messages)

            # Extract and normalize intent
            intent = response.content.strip().lower()
            self.logger.info(f"Classification result: '{intent}'")

            return intent

        except Exception as e:
            self.logger.error(f"Error during intent classification: {e}", exc_info=True)
            return "error"
