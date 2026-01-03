from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import pandas as pd

from src.shared.database import Database
from rag_logic import generate_sql, answer_with_data, load_file, schema_str, dictionary_str

load_dotenv()

app = Flask(__name__)

# Initialize DB
db = Database(
    host=os.environ["HOST"],
    port=os.environ["PORT"],
    database=os.environ["DATABASE"],
    user=os.environ["USER"],
    password=os.environ["PASSWORD"]
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run_sql", methods=["POST"])
def run_sql():
    data = request.get_json()
    sql_query = data.get("sql", "").strip()

    if not sql_query.lower().startswith("select"):
        return jsonify({"error": "Only SELECT queries are allowed."}), 400

    df = db.select_raw(sql_query)
    if df is None or df.empty:
        return jsonify({"error": "No results found or query error."}), 400

    return jsonify({
        "data": df.where(pd.notnull(df), None).to_dict(orient="records"),
        "columns": [str(c) for c in df.columns]
    })

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Step 1: Generate SQL
    sql_query = generate_sql(user_question, schema_str, dictionary_str)
    if sql_query.strip().lower() == "i am not sure on this question.":
        return jsonify({"error": "LLM could not generate a confident SQL query."}), 200

    # Step 2: Execute SQL
    df = db.select_raw(sql_query)
    if df is None or df.empty:
        return jsonify({"error": "No results found or query error."}), 200

    # Step 3: Get final answer
    final_answer = answer_with_data(df, user_question)

    return jsonify({
    "sql": sql_query,
    "data": df.where(pd.notnull(df), None).to_dict(orient="records"),
    "columns": [str(c) for c in df.columns],
    "answer": final_answer
    })

if __name__ == "__main__":
    app.run(debug=True)
