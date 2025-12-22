import cv2
import numpy as np
import easyocr
import pandas as pd
import re
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from datetime import datetime


class FormOCRProcessor:
    def __init__(self, image_path):
        self.image_path = image_path

        # EasyOCR for detection
        self.detector = easyocr.Reader(['en'], gpu=False)

        # TrOCR for handwriting recognition
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.image = None
        self.processed = None

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------
    def preprocess(self):
        self.image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 10
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # -------------------------------------------------
    # EASYOCR: DETECT TEXT REGIONS
    # -------------------------------------------------
    def detect_regions(self):
        return self.detector.readtext(self.processed)

    # -------------------------------------------------
    # TrOCR: RECOGNIZE HANDWRITING
    # -------------------------------------------------
    def recognize_with_trocr(self, roi):
        pil_img = Image.fromarray(roi).convert("RGB")
        pixel_values = self.processor(
            images=pil_img, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return text.strip()

    # -------------------------------------------------
    # GROUP INTO ROWS
    # -------------------------------------------------
    def group_rows(self, detections, threshold=30):
        rows = []

        for bbox, _, _ in sorted(detections, key=lambda x: x[0][0][1]):
            y = bbox[0][1]
            placed = False

            for row in rows:
                if abs(row["y"] - y) < threshold:
                    row["boxes"].append(bbox)
                    placed = True
                    break

            if not placed:
                rows.append({"y": y, "boxes": [bbox]})

        return rows

    # -------------------------------------------------
    # COLUMN MAPPING (FORM-SPECIFIC)
    # -------------------------------------------------
    def column_from_x(self, x):
        if x < 130:
            return "HTS Number"
        elif x < 270:
            return "Date of Visit"
        elif x < 480:
            return "Name"
        elif x < 600:
            return "Contact"
        elif x < 670:
            return "Age"
        elif x < 740:
            return "Sex"
        elif x < 840:
            return "Children Alive"
        elif x < 980:
            return "Marital Status"
        elif x < 1120:
            return "Tested Before"
        else:
            return "Test Result"

    # -------------------------------------------------
    # EXTRACT TABLE
    # -------------------------------------------------
    def extract_table(self, detections):
        rows = self.group_rows(detections)
        records = []

        for row in rows:
            record = {}

            for bbox in row["boxes"]:
                x = int(bbox[0][0])
                y = int(bbox[0][1])
                w = int(bbox[1][0] - bbox[0][0])
                h = int(bbox[2][1] - bbox[0][1])

                roi = self.image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                text = self.recognize_with_trocr(roi)
                text = re.sub(r"[^A-Za-z0-9+/]", "", text)

                column = self.column_from_x(x)
                record[column] = text

            # Strict validation
            if "HTS Number" in record and re.fullmatch(r"\d{1,2}", record["HTS Number"]):
                records.append(record)

        return records

    # -------------------------------------------------
    # MAIN PIPELINE
    # -------------------------------------------------
    def process(self):
        self.preprocess()
        detections = self.detect_regions()
        records = self.extract_table(detections)
        return pd.DataFrame(records)

    # -------------------------------------------------
    # EXPORT
    # -------------------------------------------------
    def export(self, df):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"extracted_{ts}.csv", index=False)
        df.to_excel(f"extracted_{ts}.xlsx", index=False)
        print("Files exported successfully")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    processor = FormOCRProcessor("TestImage.jpeg")
    df = processor.process()

    print("\n" + "=" * 80)
    print("EXTRACTED DATA (EasyOCR + TrOCR)")
    print("=" * 80)
    print(df)

    processor.export(df)
    print("\nTotal records:", len(df))
