# This script is a detection and classification pipeline

# import stuff
from flask import Flask, request, send_file
from PIL import Image, ExifTags
import os
from contactsnap.backend.inference.detect_classify import detect_and_label
import sys
import toga

# Setup logging (remove later)
import logging
logging.basicConfig(
    level=logging.DEBUG,  
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------------
# CONFIGURATION
# -----------------------------------

# initialize Flask app
app = Flask(__name__) 

# -----------------------------------
# HELPER FUNCTIONS
# -----------------------------------

# legacy - remove
def resource_path(relative_path):
    # Get absolute path to resource, works for dev and for briefcase-built app.
    if getattr(sys, 'frozen', False):
        # Running in a bundle
        base_path = sys._MEIPASS
    else:
        # Running in dev
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)

# sometimes the images are rotated (hardware issue, I think)
DEFAULT_ROTATE = 270    # I determined this through trial and error
def auto_orient(img):
    try:
        exif = getattr(img, '_getexif', lambda: None)()
        if exif:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == 'Orientation':
                    if value == 3:
                        img = img.rotate(180, expand=True)
                    elif value == 6:
                        img = img.rotate(270, expand=True)
                    elif value == 8:
                        img = img.rotate(90, expand=True)
                    break
        else:
            # default if no EXIF data 
            img = img.rotate(DEFAULT_ROTATE, expand=True)
    except Exception as e:
        print(f"EXIF orientation failed: {e}")
    return img #img.rotate(270, expand=True)

# -----------------------------------
# ROUTE - UPLOAD IMAGE
# -----------------------------------

# names
TEMP_UPLOAD_NAME = "input_upload.jpg"
OUTPUT_IMAGE_NAME_LABEL = "output_with_labels.jpg"

# triggered when the front end sends a POST request to /upload
@app.route('/upload', methods=['POST'])
def upload_image():

    logging.info("Upload request received.")

    # dirs
    APP_DIR = toga.App.app.paths.app
    UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
    RESULTS_DIR = os.path.join(APP_DIR, "results")
    TEMP_UPLOAD_PATH = os.path.join(UPLOAD_DIR, TEMP_UPLOAD_NAME)

    # check if the request contains a file
    if 'image' not in request.files:
        return "No image uploaded", 400
    # retrieve the file from the request
    image_file = request.files['image']
    # open the image file and convert it to RGB 
    image = Image.open(image_file.stream).convert("RGB")
    # rotate the image, if necessary
    image = auto_orient(image)  
    # save the image to the TEMP_UPLOAD_PATH
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    image.save(TEMP_UPLOAD_PATH)
    print(f"Saved uploaded image to {TEMP_UPLOAD_PATH}")
    return "Upload successful", 200

# -----------------------------------
# ROUTE - DETECT/CLASSIFIY IMAGE
# -----------------------------------

# triggered when the front end sends a GET request to /detect
@app.route('/detect', methods=['GET'])
def detect():

    logging.info("Detect request received.")

    # dirs
    APP_DIR = toga.App.app.paths.app
    UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
    RESULTS_DIR = os.path.join(APP_DIR, "results")
    TEMP_UPLOAD_PATH = os.path.join(UPLOAD_DIR, TEMP_UPLOAD_NAME)
    FINAL_OUTPUT_PATH = os.path.join(RESULTS_DIR, OUTPUT_IMAGE_NAME_LABEL)

    if not os.path.exists(TEMP_UPLOAD_PATH):
        return "No image uploaded yet", 400
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # run the detection and classification
        detect_and_label(TEMP_UPLOAD_PATH, FINAL_OUTPUT_PATH)
        return send_file(FINAL_OUTPUT_PATH, mimetype='image/jpeg'), 200
    except Exception as e:
        logging.error(f"Detection failed: {e}", exc_info=True)
        return f"Detection failed: {e}", 500

# debubging 
@app.route('/ping', methods=['GET'])
def ping():
    logging.info("Ping received.")
    return "pong", 200

# -----------------------------------
# LAUCHER
# -----------------------------------
#if __name__ == '__main__':
#    app.run(host="0.0.0.0", port=5050, debug=True)
#if __name__ == '__main__':
#    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False)
