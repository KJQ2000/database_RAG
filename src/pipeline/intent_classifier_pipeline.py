from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

import yaml
from src.shared.constant import *


class IntentClassifier:
    def __init__(self, intent_system_prompt: str):
        """
        Initializes the IntentClassifier with the required configurations.
        
        :param intent_system_prompt: The system prompt used for intent classification.
        :param api_key: The OpenAI API key to access the model.
        :param model: The model to use (default is "gpt-4o-mini").
        :param temperature: The sampling temperature (default is 0 for deterministic results).
        :param max_tokens: Maximum tokens for the model's response (default is 10).
        """
        self.intent_system_prompt = intent_system_prompt
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = INTENT_LLM
        self.temperature = INTENT_TEMPERATURE
        self.max_tokens = INTENT_MAX_TOKEN

        # Initialize the LLM model
        self.intent_llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=self.api_key
        )

    def classify_intent(self, user_question: str) -> str:
        """
        Classifies the user's question into one of the predefined intents.

        :param user_question: The question input by the user.
        :return: The intent classification result.
        """
        messages = [
            SystemMessage(content=self.intent_system_prompt),
            HumanMessage(content=user_question)
        ]
        response = self.intent_llm(messages)
        return response.content.strip().lower()