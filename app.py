from flask import Flask, request, jsonify
from main import main
import time
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question'"}), 400

    question = data["question"]

    start = time.time()
    answer = main(question)
    elapsed = round(time.time() - start, 2)

    return jsonify({
        "answer": answer,
        "latency_sec": elapsed
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
