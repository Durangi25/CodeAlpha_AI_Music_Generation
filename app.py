import os
from flask import Flask, render_template, request, send_from_directory
from generate_music import generate_midi_file


app = Flask(__name__)

GENERATED_FOLDER = "generated"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        output_length = int(request.form.get("length", 300))
        temperature = float(request.form.get("temperature", 0.8))

        output_file = generate_midi_file(output_length, temperature)
        filename = os.path.basename(output_file)

        return render_template(
            "index.html",
            success=True,
            filename=filename,
            message="Music generated successfully!"
        )

    except Exception as e:
        return render_template(
            "index.html",
            success=False,
            error=str(e)
        )


@app.route("/generated/<filename>")
def download_file(filename):
    return send_from_directory(GENERATED_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)