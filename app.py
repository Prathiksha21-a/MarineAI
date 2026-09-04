from flask import Flask, render_template, request, jsonify, redirect, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os, json, cv2, time
from datetime import datetime

from detector import analyze_image
from gradcam import generate_gradcam
from screen_stream import stream

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"
HISTORY_FILE = "history/history.json"

for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER, "history"):
    os.makedirs(folder, exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)


@app.route("/")
def home():
    return render_template("index.html", active="home")


@app.route("/detection")
def detection():
    return render_template("detection.html", active="detection")


@app.route("/live_detection")
def live_detection():
    return render_template("live_detection.html", active="live")


@app.route("/xai")
def xai():
    return render_template("xai.html", active="xai")


@app.route("/about")
def about():
    return render_template("about.html", active="about")


@app.route("/history")
def history():
    with open(HISTORY_FILE) as f:
        data = json.load(f)

    return render_template(
        "history.html",
        history=list(reversed(data)),
        active="history"
    )


def save_history(filename, result, original):
    with open(HISTORY_FILE) as f:
        history = json.load(f)

    history.append(
        {
            "filename": filename,
            "date": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "original_image": original,
            "output_image": result["output_image"],
            **{
                k: result[k]
                for k in (
                    "fish_present",
                    "fish_abundance",
                    "large_fish",
                    "species"
                )
            },
        }
    )

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def process_upload(file, record=True):

    name = (
        f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_"
        f"{secure_filename(file.filename or 'capture.jpg')}"
    )

    input_path = os.path.join(
        UPLOAD_FOLDER,
        name
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "result_" + name
    )

    if hasattr(file, "save"):
      file.save(input_path)
    else:
      with open(input_path, "wb") as f:
        f.write(file.read())

    result = analyze_image(
        input_path,
        output_path
    )

    result["output_image"] = output_path.replace("\\", "/")
    result["original_image"] = input_path.replace("\\", "/")


    if record:
        save_history(
            name,
            result,
            result["original_image"]
        )

    return result



@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify(error="No image uploaded"),400

    return jsonify(
        process_upload(
            request.files["image"]
        )
    )



@app.route("/live_detect", methods=["POST"])
def live_detect():

    frame = request.files.get("image")

    if not frame:
        return jsonify(
            error="No frame received"
        ),400

    return jsonify(
        process_upload(
            frame,
            record=False
        )
    )



@app.route("/phone_stream")
def phone_stream():
    def generate():
        while True:
            success, frame = stream.read()
            if not success or frame is None:
                time.sleep(0.2)
                continue
            ret, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
            time.sleep(0.04)

    return Response(
        generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )



@app.route("/xai_predict", methods=["POST"])
def xai_predict():

    image = request.files.get("image")

    if not image:
        return jsonify(error="No image uploaded"),400


    name = (
        f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_"
        f"{secure_filename(image.filename)}"
    )


    input_path = os.path.join(
        UPLOAD_FOLDER,
        name
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "xai_" + name
    )


    image.save(input_path)

    generate_gradcam(
        input_path,
        output_path
    )


    return jsonify(
        original_image=input_path.replace("\\","/"),
        xai_image=output_path.replace("\\","/")
    )



@app.route("/clear_history")
def clear_history():

    with open(HISTORY_FILE,"w") as f:
        json.dump([],f)

    return redirect("/history")


@app.route("/phone_detect", methods=["POST"])
def phone_detect():

    success, frame = stream.read()

    if not success:
        return jsonify(error=stream.last_error or "Phone screen not found"), 400


    filename = f"phone_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"

    input_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "result_" + filename
    )


    cv2.imwrite(
        input_path,
        frame
    )


    result = analyze_image(
        input_path,
        output_path
    )


    result["output_image"] = output_path.replace("\\", "/")
    result["original_image"] = input_path.replace("\\", "/")


    return jsonify(result)

if __name__=="__main__":
    app.run(
        debug=True
    )