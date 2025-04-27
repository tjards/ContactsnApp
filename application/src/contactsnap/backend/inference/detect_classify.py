# this is the digit detection module (using yolov8)

# --------------------------------------------------------------------
# IMPORT STUFF
# --------------------------------------------------------------------

import os
import cv2
from ultralytics import YOLO
import yaml

from contactsnap.backend.utils.custom_filters import compute_iou, iou_threshold, filter_digit_rows

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------

MODEL_NAME              = 'handwriting_cleaned_subset_002'
#MODEL_NAME              = 'digit_detector_yolo'
CONFIDENCE_THRESHOLD    = 0.2 # nominally, very small

# base directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# optionals
POSTPROCESS = True

# --------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------

# just draw boxes on the image (optional)
def draw_boxes(image_name):

    # Get path to image
    image_path = os.path.join(BASE_DIR, "uploads", image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # run YOLO digit detection
    digit_imgs, boxes, image, _, _, _ = detect_and_classify(image_path)

    # draw boxes and save visualization
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)  # make sure it exists
    output_path = os.path.join(results_dir, "output_with_boxes.jpg")

    cv2.imwrite(output_path, image)

# post-process the results
def postprocess_results(confidences, boxes, digit_imgs, yolo_classes):
    
    #filter overlapping boxes
    filtered = sorted(zip(confidences, boxes, digit_imgs, yolo_classes), reverse=True)
    final_boxes, final_imgs, final_classes, final_confs = [], [], [], []
    for conf, box, img, cls in filtered:
        if all(compute_iou(box, kept_box) < iou_threshold for kept_box in final_boxes):
            final_boxes.append(box)
            final_imgs.append(img)
            final_classes.append(cls)
            final_confs.append(conf)

    # filter out rows with too few digits
    final_boxes, final_imgs, final_classes, final_confs = filter_digit_rows(
    final_boxes, final_imgs, final_classes, final_confs,
    min_digits=3,
    max_vertical_diff=100,
    max_horizontal_gap=5000
    )

    return final_boxes, final_imgs, final_classes, final_confs

# --------------------------------------------------------------------
# DETECT AND CLASSIFY FUNCTION
# --------------------------------------------------------------------

def detect_and_classify(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # load model (assumes yolov8)
    model_path = os.path.join(BASE_DIR, "inference", "models", MODEL_NAME, "weights", "best.pt")
    model = YOLO(model_path)

    # load the class map
    yaml_path = os.path.join(BASE_DIR, "inference", "models", MODEL_NAME, "data.yaml") # I added this from the dataset
    with open(yaml_path, "r") as f:
        class_map = yaml.safe_load(f)["names"]

    # load image
    img_color = cv2.imread(image_path)

    # run inference directly on image array 
    results = model.predict(source=img_color, conf=CONFIDENCE_THRESHOLD, save=False)

    # initialize
    digit_imgs      = []    # list of digit images
    bounding_boxes  = []    # list of bounding boxes
    yolo_classes    = []    # list of YOLO class predictions
    confidences     = []    # list of YOLO confidence scores

    for box in results[0].boxes:

        # find bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        bounding_boxes.append((x1, y1, w, h))

        # crop and resize digit region
        crop = img_color[y1:y2, x1:x2]
        resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
        digit_imgs.append(resized)

        # get YOLO class prediction
        class_id = int(box.cls[0])
        yolo_classes.append(class_id)

        # get YOLO confidence score
        confidence = float(box.conf[0])
        confidences.append(confidence)

    return digit_imgs, bounding_boxes, img_color, yolo_classes, class_map, confidences

# --------------------------------------------------------------------
# LABELING FUNCTION - compares to another classifier (optional)
# --------------------------------------------------------------------

# detect and label digits in an image 
def detect_and_label(image_path, output_path, include_yolo_classes=True, include_cnn_classes=False):

    # run the detect and classify function (above)
    digit_imgs, boxes, _, yolo_classes, yolo_classes_map, confidences = detect_and_classify(image_path)

    # do some post-processing
    if POSTPROCESS:
        final_boxes, final_imgs, final_classes, final_confs = postprocess_results(confidences, boxes, digit_imgs, yolo_classes)
    else:
         final_confs, final_boxes, final_imgs, final_classes = confidences, boxes, digit_imgs, yolo_classes


    # classify with another model (for compare)
    if include_cnn_classes:
        print("This module is not yet implemented.")
        #cnn_predictions, cnn_classes_map = classify_digits(final_imgs)
    else:
        cnn_predictions = [None] * len(final_imgs)
        cnn_classes_map = None

    # read the original image for labeling
    img = cv2.imread(image_path)

    # draw define labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    font_thickness = 10
    color_conf = (0, 0, 255)
    color_label = (255, 0, 0)
    color_box = (0, 255, 0)
    box_thickness = 5  

    # draw the bounding boxes
    for (x, y, w, h) in final_boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), color_box, box_thickness)

    #flat_boxes = [box for row in final_boxes for box in row]
    
    # Now draw the boxes
    #for (x, y, w, h) in flat_boxes:
    #    cv2.rectangle(img, (x, y), (x + w, y + h), color_box, box_thickness)


    # draw the label(s)
    for cnn_id, yolo_id, (x, y, w, h), conf in zip(cnn_predictions, final_classes, final_boxes, final_confs):
        
        yolo_label = yolo_classes_map[yolo_id]
        conf_str = f"{conf:.2f}"

        if include_cnn_classes:
            cnn_label = cnn_classes_map[cnn_id]
            cv2.putText(img, str(cnn_label), 
                        (x, y - 10 if y - 10 > 10 else y + h + 20), 
                        font, font_scale, color_conf, font_thickness)

        if include_yolo_classes:
            cv2.putText(img, str(conf_str), 
                        (x, y - 170 if y - 70 > 10 else y + h + 60), 
                        font, font_scale * 0.5, color_conf, 2)
            cv2.putText(img, str(yolo_label), 
                        (x, y - 40 if y - 40 > 10 else y + h + 40), 
                        font, font_scale * 0.8, color_label, font_thickness)

    # save the labeled image
    cv2.imwrite(output_path, img)