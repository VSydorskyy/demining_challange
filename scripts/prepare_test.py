import os
import cv2
import numpy as np
from pathlib import Path
import argparse


def load_yolo_labels(label_path):
    """Load YOLO-format labels from a .txt file."""
    boxes = []
    if Path(label_path).exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = map(float, parts)
                boxes.append((int(cls), xc, yc, w, h))
    return boxes


def convert_to_absolute(box, img_width, img_height):
    """Convert normalized YOLO box to absolute pixel coordinates."""
    cls, xc, yc, w, h = box
    abs_w, abs_h = w * img_width, h * img_height
    x1 = xc * img_width - abs_w / 2
    y1 = yc * img_height - abs_h / 2
    x2 = x1 + abs_w
    y2 = y1 + abs_h
    return cls, x1, y1, x2, y2


def convert_to_yolo_format(cls, x1, y1, x2, y2, img_width, img_height):
    """Convert absolute box back to YOLO normalized format."""
    x1 = max(0, min(x1, img_width))
    x2 = max(0, min(x2, img_width))
    y1 = max(0, min(y1, img_height))
    y2 = max(0, min(y2, img_height))
    xc = (x1 + x2) / 2 / img_width
    yc = (y1 + y2) / 2 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return cls, xc, yc, w, h


def find_max_density_crop(boxes, img_w, img_h, crop_size=640):
    """Find the crop with the most bounding boxes intersecting it."""
    step = crop_size // 2
    max_count = 0
    best_crop = (0, 0)

    for y in range(0, max(1, img_h - crop_size + 1), step):
        for x in range(0, max(1, img_w - crop_size + 1), step):
            x1_crop, y1_crop = x, y
            x2_crop, y2_crop = x + crop_size, y + crop_size
            count = 0
            for _, x1, y1, x2, y2 in boxes:
                if x2 > x1_crop and x1 < x2_crop and y2 > y1_crop and y1 < y2_crop:
                    count += 1
            if count > max_count:
                max_count = count
                best_crop = (x1_crop, y1_crop)

    return best_crop


def crop_image_and_labels(image_path, label_path, out_img_path, out_label_path, crop_size=640):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return

    h, w = image.shape[:2]
    boxes = load_yolo_labels(label_path)
    abs_boxes = [convert_to_absolute(box, w, h) for box in boxes]

    if len(abs_boxes) == 0:
        return  # skip if no boxes

    cx, cy = find_max_density_crop(abs_boxes, w, h, crop_size)
    crop = image[cy:cy + crop_size, cx:cx + crop_size]
    cv2.imwrite(str(out_img_path), crop)

    new_labels = []
    for cls, x1, y1, x2, y2 in abs_boxes:
        x1_new = x1 - cx
        x2_new = x2 - cx
        y1_new = y1 - cy
        y2_new = y2 - cy

        if x2_new <= 0 or x1_new >= crop_size or y2_new <= 0 or y1_new >= crop_size:
            continue

        x1_new = max(0, min(x1_new, crop_size))
        x2_new = max(0, min(x2_new, crop_size))
        y1_new = max(0, min(y1_new, crop_size))
        y2_new = max(0, min(y2_new, crop_size))

        yolo_box = convert_to_yolo_format(cls, x1_new, y1_new, x2_new, y2_new, crop_size, crop_size)
        new_labels.append(yolo_box)

    with open(out_label_path, "w") as f:
        for cls, xc, yc, w_, h_ in new_labels:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w_:.6f} {h_:.6f}\n")


def process_dataset(images_dir, labels_dir, output_dir="test", crop_size=640):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for img_path in images_dir.glob("*.jp*"):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        out_img = out_images / f"{stem}.jpg"
        out_lbl = out_labels / f"{stem}.txt"
        crop_image_and_labels(img_path, label_path, out_img, out_lbl, crop_size)

    print(f"Processed dataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop images to 640x640 around regions with max bounding boxes.")
    parser.add_argument("--images", type=str, required=True, help="Path to folder with images.")
    parser.add_argument("--labels", type=str, required=True, help="Path to folder with YOLO COCO8 .txt labels.")
    parser.add_argument("--output", type=str, default="test", help="Output folder for cropped images and labels.")
    parser.add_argument("--size", type=int, default=640, help="Crop size (default: 640).")
    args = parser.parse_args()

    process_dataset(args.images, args.labels, output_dir=args.output, crop_size=args.size)
