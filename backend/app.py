import os
import glob
import cv2
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__, template_folder="templates")

# ESP32-CAM Stream URL
ESP32_URL = "http://192.168.1.3/capture"  # Replace with your ESP32-CAM IP

# Load YOLOv8 Weapon Detection Model
model = YOLO("best.pt")  # Ensure the model is in the backend folder

# Define folders for uploaded and processed images
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

@app.route("/")
def index():
    """Serve the frontend page."""
    return render_template("index.html")

def capture_image():
    """Fetches an image from the ESP32-CAM."""
    try:
        response = requests.get(ESP32_URL, timeout=5)
        if response.status_code == 200:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp
            filename = f"frame_{timestamp}.jpg"
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            with open(image_path, "wb") as file:
                file.write(response.content)
            return filename
        else:
            print("❌ Error: Failed to fetch image from ESP32-CAM")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Exception: {e}")
        return None

def process_image(image_path, filename):
    """Runs YOLOv8 weapon detection on the captured image and saves the processed image."""
    print(f"🔍 Processing Image: {image_path}")

    results = model.predict(image_path, save=True, conf=0.25)  # Adjust confidence threshold

    # Find the latest YOLOv8 output directory
    runs_dir = "runs/detect/"
    latest_folder = max(glob.glob(os.path.join(runs_dir, "predict*")), key=os.path.getmtime)

    # Get the latest processed image
    processed_images = sorted(glob.glob(os.path.join(latest_folder, "*.jpg")), key=os.path.getmtime)

    if not processed_images:
        print("❌ No processed images found.")
        return None, False, 0.0

    yolo_output_path = processed_images[-1]  # Get the latest processed image
    print(f"✅ Found Processed Image: {yolo_output_path}")

    # Create a unique filename for processed images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_filename = f"processed_{timestamp}.jpg"
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)

    # Move YOLOv8 processed image to `static/processed/`
    os.rename(yolo_output_path, processed_path)

    weapon_detected = False
    confidence = 0.0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        conf = round(float(box.conf[0]) * 100, 2)

        if class_id == 0:  # Assuming class ID 0 is "Weapon"
            weapon_detected = True
            confidence = max(confidence, conf)  # Use highest confidence

    return processed_filename, weapon_detected, confidence

@app.route("/process", methods=["GET"])
def process_image_from_esp32():
    """Fetch an image from ESP32-CAM, process it, and return results."""
    filename = capture_image()
    if not filename:
        return jsonify({"error": "Failed to capture image from ESP32-CAM"}), 500

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    processed_filename, weapon_detected, confidence = process_image(image_path, filename)

    if not processed_filename:
        return jsonify({"error": "Failed to process image"}), 500

    processed_image_url = url_for("get_processed_image", filename=processed_filename, _external=True)

    return jsonify({
        "weapon_detected": weapon_detected,
        "confidence": confidence,
        "processed_image": processed_image_url
    })

@app.route("/processed/<filename>")
def get_processed_image(filename):
    """Serve processed images."""
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)





























# import os
# import cv2
# import numpy as np
# import requests
# from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
# from ultralytics import YOLO
# from datetime import datetime

# app = Flask(__name__, template_folder="templates")

# # ESP32-CAM Stream URL
# ESP32_URL = "http://192.168.1.3/capture"  # Replace with your ESP32-CAM IP

# # Load YOLOv8 Weapon Detection Model
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
# model = YOLO(MODEL_PATH)

# # Run prediction on an image (Replace 'test_image.jpg' with your actual image)
# # results = model.predict("test_image.jpg", save=True)

# # Define folders for uploaded and processed images
# UPLOAD_FOLDER = "static/uploads"
# PROCESSED_FOLDER = "static/processed"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# @app.route("/")
# def index():
#     """Serve the frontend page."""
#     return render_template("index.html")

# def capture_image():
#     """Fetches an image from the ESP32-CAM."""
#     try:
#         response = requests.get(ESP32_URL, timeout=5)
#         if response.status_code == 200:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp
#             filename = f"frame_{timestamp}.jpg"
#             image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             with open(image_path, "wb") as file:
#                 file.write(response.content)
#             return filename
#         else:
#             return None
#     except requests.exceptions.RequestException:
#         return None

# def process_image(image_path, filename):
#     """Runs YOLOv8 weapon detection on the captured image."""
#     results = model.predict(image_path, save=True)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique filename
#     processed_filename = f"processed_{timestamp}.jpg"
#     processed_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)

#     # Find the processed image in 'runs/detect/predict/' and move it
#     yolo_output_dir = os.path.join("runs", "detect", "predict")
#     yolo_output_path = os.path.join(yolo_output_dir, filename)

#     if os.path.exists(yolo_output_path):
#         os.rename(yolo_output_path, processed_path)  # Move the processed image

#     weapon_detected = False
#     confidence = 0.0

#     for box in results[0].boxes:
#         class_id = int(box.cls[0])
#         conf = round(float(box.conf[0]) * 100, 2)

#         if class_id == 0:  # Assuming class ID 0 is "Weapon"
#             weapon_detected = True
#             confidence = max(confidence, conf)  # Use highest confidence

#     return processed_filename, weapon_detected, confidence

# # def process_image(image_path, filename):
# #     """Runs YOLOv8 weapon detection on the captured image."""
# #     results = model.predict(image_path, save=True)

# #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp
# #     processed_filename = f"processed_{timestamp}.jpg"
# #     processed_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)

# #     # Move YOLO's processed image to our processed folder
# #     yolo_output_path = os.path.join("runs", "detect", "predict", filename)

# #     if os.path.exists(yolo_output_path):
# #         if os.path.exists(processed_path):  
# #             os.remove(processed_path)  # Delete existing file if necessary
# #         os.rename(yolo_output_path, processed_path)  # Move the new processed image

# #     weapon_detected = False
# #     confidence = 0.0

# #     for box in results[0].boxes:
# #         class_id = int(box.cls[0])
# #         conf = round(float(box.conf[0]) * 100, 2)

# #         if class_id == 0:  # Assuming class ID 0 is "Weapon"
# #             weapon_detected = True
# #             confidence = max(confidence, conf)  # Use highest confidence

# #     return processed_filename, weapon_detected, confidence

# @app.route("/process", methods=["GET"])
# def process_image_from_esp32():
#     """Fetch an image from ESP32-CAM, process it, and return results."""
#     filename = capture_image()
#     if not filename:
#         return jsonify({"error": "Failed to capture image from ESP32-CAM"}), 500

#     image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#     processed_filename, weapon_detected, confidence = process_image(image_path, filename)

#     processed_image_url = url_for("get_processed_image", filename=processed_filename, _external=True)

#     return jsonify({
#         "weapon_detected": weapon_detected,
#         "confidence": confidence,
#         "processed_image": processed_image_url
#     })

# @app.route("/processed/<filename>")
# def get_processed_image(filename):
#     """Serve processed images."""
#     return send_from_directory(app.config["PROCESSED_FOLDER"], filename)

# if __name__ == "__main__":
#     app.run(debug=True)
