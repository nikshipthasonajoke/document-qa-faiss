import os
from flask import Flask, render_template, request
from retriever import Retriever
from ingest_runtime import build_index_from_file

UPLOAD_FOLDER = "../data/uploads"

if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__, template_folder="../templates")
retriever = Retriever()

@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    final_answer = ""
    supporting_context = []
    message = ""

    # Document upload
    if "document" in request.files:
        file = request.files["document"]
        if file.filename:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            build_index_from_file(file_path)
            retriever.load()
            message = "Document indexed successfully."

    # Question handling
    if request.method == "POST" and "question" in request.form:
        question = request.form["question"]
        results = retriever.search(question, k=3)

        if results:
            final_answer = results[0]
            supporting_context = results[1:]
        else:
            final_answer = (
                "The uploaded document does not contain information "
                "to answer this question."
            )

    return render_template(
        "index.html",
        question=question,
        final_answer=final_answer,
        supporting_context=supporting_context,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)
