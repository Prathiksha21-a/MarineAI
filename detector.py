from ultralytics import YOLO
import cv2

fish_model = YOLO("models/fish_best.pt")
large_fish_model = YOLO("models/largefish_best.pt")
FISH_CONF = 0.45
# Species labels are only shown for high-confidence large-fish predictions.
# This avoids turning an ordinary fish into a Shark/Tuna false positive.
LARGE_FISH_CONF = 0.65
FISH_IMGSZ = 960
FISH_IOU = 0.6


def calculate_density(count):
    if count == 0:
        return "Low"
    if count <= 5:
        return "Low"
    if count <= 15:
        return "Medium"
    return "High"


def _class_name(results, box):
    names = results[0].names
    class_id = int(box.cls[0])
    return (
        str(names.get(class_id, ""))
        if isinstance(names, dict)
        else str(names[class_id])
    )


def _large_species_boxes(results):
    return [
        box
        for box in results[0].boxes
        if _class_name(results, box).lower() in {"shark", "tuna"}
    ]


def _species_from_boxes(results, boxes, fallback="-"):
    if not boxes:
        return fallback
    best_box = max(boxes, key=lambda box: float(box.conf[0]))
    return _class_name(results, best_box).title()


def _iou(first, second):
    """Return overlap between two [x1, y1, x2, y2] boxes."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0


def _smaller_box_overlap(first, second):
    """How much of the smaller box is covered by the other box."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    smallest_area = min(first_area, second_area)
    return intersection / smallest_area if smallest_area else 0


def _draw_fish_boxes(image, fish_boxes, large_boxes):
    """Draw generic fish only when no large-fish box describes the same animal."""
    for box in fish_boxes:
        coordinates = [int(value) for value in box.xyxy[0].tolist()]
        center_x = (coordinates[0] + coordinates[2]) / 2
        center_y = (coordinates[1] + coordinates[3]) / 2
        same_large_fish = any(
            _smaller_box_overlap(coordinates, large_box) >= 0.20
            or (
                large_box[0] <= center_x <= large_box[2]
                and large_box[1] <= center_y <= large_box[3]
            )
            for large_box in large_boxes
        )
        if same_large_fish:
            continue
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 150, 0), 2)
        label = f"fish {confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 150, 0),
            2,
        )
    return image


def _draw_large_species_boxes(image, results, boxes):
    for box in boxes:
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        label = f"{_class_name(results, box)} {float(box.conf[0]):.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 190, 90), 2)
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 190, 90),
            2,
        )
    return image


def analyze_image(input_path, output_path):
    fish_results = fish_model.predict(
        input_path,
        conf=FISH_CONF,
        imgsz=FISH_IMGSZ,
        iou=FISH_IOU,
        verbose=False,
    )
    large_results = large_fish_model.predict(
        input_path, conf=LARGE_FISH_CONF, verbose=False
    )
    fish_count = len(fish_results[0].boxes)
    species_boxes = _large_species_boxes(large_results)
    large_count = len(species_boxes)
    image = cv2.imread(input_path)
    large_boxes = [box.xyxy[0].tolist() for box in species_boxes]
    image = _draw_fish_boxes(image, fish_results[0].boxes, large_boxes)
    if large_count:
        # Large-fish labels replace overlapping generic fish labels.
        image = _draw_large_species_boxes(image, large_results, species_boxes)
    cv2.imwrite(output_path, image)
    return {
        "fish_present": "Yes" if fish_count else "No",
        "fish_abundance": calculate_density(fish_count),
        "fish_density": calculate_density(fish_count),
        "large_fish": "Yes" if large_count else "No",
        "species": _species_from_boxes(
            large_results,
            species_boxes,
            fallback="Fish" if fish_count else "-",
        ),
        "confidence": "-",
        "recommendation": "Analysis complete.",
    }