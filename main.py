from dotenv import load_dotenv
import os
import pandas as pd
import argparse

from src.shared.database import Database
from src.pipeline.intent_classifier_pipeline import IntentClassifier
from src.shared.utils import load_config

load_dotenv()

current_path = os.getcwd()

config_file_path = os.path.join(current_path,'config','config.yaml')
configs = load_config(config_file_path)


parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument('--intent_classifier', type=str, help='Intent Classifier to classify user questiuons', required=False)

args = parser.parse_args()


if args.intent_classifier:
    intent_system_prompt = configs['intent_system_prompt']

    classifier = IntentClassifier(
        intent_system_prompt=intent_system_prompt
    )
    
    user_question = "What is your return policy?"
    intent = classifier.classify_intent(user_question)
    print(f"Classified intent: {intent}")