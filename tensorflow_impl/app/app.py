from flask import Flask, render_template, request
import os
from deeplearning import OCR
from datetime import datetime

# webserver gateway interface
app = Flask(__name__)
app.config["DEBUG"] = True
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, "static/upload/")


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        upload_file = request.files["image_name"]
        model_selected = request.form.get("model_select")
        print("Selected extraction model:", model_selected)
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        text = OCR(path_save, filename, model_selected)

        return render_template(
            "index.html",
            upload=True,
            upload_image=filename,
            text=text,
            current_year=datetime.now().year,
        )

    return render_template("index.html", upload=False, current_year=datetime.now().year)


if __name__ == "__main__":
    app.run(port=5001)
