
# --------------------------------------------------------------------
# Intersection over union (IOU)
# --------------------------------------------------------------------

'''
This is a measure of overlap between two bounding boxes

+--------+
| boxA   |
|    +----------------+
|    | overlap        |
+----|----------------+
     |        boxB     |
     +-----------------+
'''

# how much overlap is allowed
iou_threshold = 0.3 #  1 = perfect overlaop, 0 = no overlap

def compute_iou(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    xA2, yA2 = xA + wA, yA + hA
    xB2, yB2 = xB + wB, yB + hB
    xL = max(xA, xB)
    yT = max(yA, yB)
    xR = min(xA2, xB2)
    yB = min(yA2, yB2)
    if xR < xL or yB < yT:
        return 0.0
    interArea = (xR - xL) * (yB - yT)
    unionArea = wA * hA + wB * hB - interArea
    return interArea / unionArea if unionArea > 0 else 0.0

def filter_digit_rows(boxes, imgs, classes, confs, min_digits=4, max_vertical_diff=50, max_horizontal_gap=400):
    """
    Filters bounding boxes to find all sequences of horizontally aligned digits.

    Args:
        boxes (list of tuples): (x, y, w, h)
        imgs (list): digit images
        classes (list): predicted classes
        confs (list): predicted confidences
        min_digits (int): minimum number of digits to count as a valid row
        max_vertical_diff (int): maximum vertical misalignment
        max_horizontal_gap (int): maximum allowed gap between adjacent digits

    Returns:
        Filtered lists: boxes, imgs, classes, confs
    """

    if not boxes:
        return [], [], [], []

    # Sort everything left to right
    combined = sorted(zip(boxes, imgs, classes, confs), key=lambda x: x[0][0])

    rows = []
    current_row = [combined[0]]

    for item in combined[1:]:
        prev_box = current_row[-1][0]
        curr_box = item[0]

        # Vertical center alignment
        prev_center_y = prev_box[1] + prev_box[3] / 2
        curr_center_y = curr_box[1] + curr_box[3] / 2
        vertical_diff = abs(prev_center_y - curr_center_y)

        # Horizontal spacing
        prev_right = prev_box[0] + prev_box[2]
        curr_left = curr_box[0]
        horizontal_dist = curr_left - prev_right

        # Belongs to same row?
        if vertical_diff <= max_vertical_diff and horizontal_dist <= max_horizontal_gap:
            current_row.append(item)
        else:
            if len(current_row) >= min_digits:
                rows.append(current_row)
            current_row = [item]  # start a new group

    # Check the last row
    if len(current_row) >= min_digits:
        rows.append(current_row)

    if not rows:
        return [], [], [], []

    # Flatten all valid rows 
    flattened = [item for row in rows for item in row]

    # Unpack back into lists 
    final_boxes, final_imgs, final_classes, final_confs = zip(*flattened)

    return list(final_boxes), list(final_imgs), list(final_classes), list(final_confs)
