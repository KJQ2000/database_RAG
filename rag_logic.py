import os
from dotenv import load_dotenv
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from constant import SCHEMA, DICT_DIR

load_dotenv()

def load_file(path):
    """Load text file content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=512,
    verbose=False,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

system_template = """
You are a PostgreSQL expert for a jewelry store.
Given the database schema:

{schema}

And the meaning of each table and column:

{dictionary}

Generate ONLY a valid PostgreSQL SELECT SQL query to answer the user question.
pls add schema name before table name. Example: SELECT * FROM konghin.stock.
Do NOT use double quotes ("") around schema, table names, or column names.
Use lowercase for schema and table names.
Do not add explanations or markdown, just output the SQL.
If you are not sure about the question, just output 'I am not sure on this question.'
"""
human_template = "{user_question}"

def build_prompt(schema: str, dictionary: str, user_question: str):
    system_msg = SystemMessagePromptTemplate.from_template(system_template)
    human_msg = HumanMessagePromptTemplate.from_template(human_template)
    prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
    return prompt.format_prompt(schema=schema, dictionary=dictionary, user_question=user_question).to_messages()

def generate_sql(user_question: str, schema: str, dictionary: str) -> str:
    messages = build_prompt(schema, dictionary, user_question)
    response = llm(messages)
    print("\nGenerated SQL:\n")
    print(response.content)
    return response.content

def answer_with_data(df: pd.DataFrame, user_question: str) -> str:
    data_str = df.to_csv(index=False)
    analysis_prompt = f"""
    You are a data analyst for a jewelry store. 
    The user asked: "{user_question}"
    Here is the query result:
    {data_str}

    Based on this data, provide a concise and accurate answer.
    If user ask in mandarin, pls reply in mandarin else answer in english.
    pls answer in a structured format. If table needed, answer in table, and text remains text.
    """
    messages = [HumanMessagePromptTemplate.from_template(analysis_prompt).format()]
    response = llm(messages)
    return response.content

schema_str = SCHEMA
dictionary_str = load_file(DICT_DIR)
