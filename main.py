from dotenv import load_dotenv
import os
import pandas as pd
import argparse

from src.shared.database import Database
from src.pipeline.intent_classifier_pipeline import IntentClassifier
from src.pipeline.build_vector_db_pipeline import BuildVectorDB
from src.shared.utils import load_config
from src.shared.constant import *

# python main.py --intent_classifier

load_dotenv()

def main(query:str):

    current_path = os.getcwd()

    config_file_path = os.path.join(current_path,'config','config.yaml')
    configs = load_config(config_file_path)


    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument('--intent_classifier',action='store_true', help='Run the intent classifier pipeline')
    parser.add_argument('--answer_question',action='store_true', help='Pipeline for anaswering question')

    args = parser.parse_args()


    if args.intent_classifier:
        intent_system_prompt = configs['intent_system_prompt']

        classifier = IntentClassifier(intent_system_prompt=intent_system_prompt)
        
        intent = classifier.classify_intent(query)
        return intent
    
    if args.answer_question:
        persist_path = os.path.join(current_path, DB_RAG_VECTOR_DB_DIR)
        vb = BuildVectorDB(persist_path=persist_path,embedding_model="text-embedding-3-small")
        vb.build_from_knowledge_base(knowledge_file=KNOWLEDGE_BASE_PATH,rebuild=True)
        

if __name__ == "__main__":
    user_question = "What is your return policy?"
    print(main(query = user_question))