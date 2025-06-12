import os
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the smaller model to stay under memory limits
qa_pipeline = pipeline(
    "question-answering",
    model="sshleifer/tiny-distilbert-base-cased-distilled-squad"
)

@app.route("/qa", methods=["POST"])
def qa():
    data = request.get_json()
    question = data.get("question")
    context = data.get("context")
    if not question or not context:
        return jsonify({"error": "Missing question or context"}), 400
    result = qa_pipeline(question=question, context=context)
    return jsonify({"answer": result["answer"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
