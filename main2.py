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

        # EasyOCR for text detection (layout)
        self.detector = easyocr.Reader(['en'], gpu=False)

        # TrOCR for handwriting recognition
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten",
           # use_fast=True
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.image = None
        self.processed = None

    # -------------------------------------------------
    # PREPROCESS IMAGE
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
        print(" Detecting text regions with EasyOCR...")
        detections = self.detector.readtext(self.processed)
        print(f" Detected {len(detections)} text regions\n")
        return detections

    # -------------------------------------------------
    # TrOCR: HANDWRITING RECOGNITION
    # -------------------------------------------------
    def recognize_with_trocr(self, roi):
        pil_img = Image.fromarray(roi).convert("RGB")

        pixel_values = self.processor(
            images=pil_img,
            return_tensors="pt",
           # use_fast=True
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=16
            )

        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    # -------------------------------------------------
    # GROUP DETECTIONS INTO ROWS
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
    # MAP X POSITION TO FORM COLUMN
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
    # EXTRACT TABLE DATA
    # -------------------------------------------------
    def extract_table(self, detections):
        rows = self.group_rows(detections)
        records = []

        print(f" Processing {len(rows)} detected rows...\n")

        for i, row in enumerate(rows):
            print(f" Row {i + 1}/{len(rows)}")
            record = {}

            for bbox in row["boxes"]:
                x = int(bbox[0][0])
                y = int(bbox[0][1])
                w = int(bbox[1][0] - bbox[0][0])
                h = int(bbox[2][1] - bbox[0][1])

                roi = self.image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                column = self.column_from_x(x)

                # Skip small header-like boxes
                if h < 25:
                    continue

                print(f"   TrOCR → {column}")

                text = self.recognize_with_trocr(roi)
                text = re.sub(r"[^A-Za-z0-9+/]", "", text)

                record[column] = text

            # Keep only valid data rows
            if "HTS Number" in record and re.fullmatch(r"\d{1,2}", record["HTS Number"]):
                records.append(record)

        return records

    # -------------------------------------------------
    # MAIN PIPELINE
    # -------------------------------------------------
    def process(self):
        print("\n Starting OCR pipeline...\n")
        self.preprocess()
        detections = self.detect_regions()
        records = self.extract_table(detections)
        return pd.DataFrame(records)

    # -------------------------------------------------
    # EXPORT RESULTS
    # -------------------------------------------------
    def export(self, df):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"extracted_form_{ts}.csv"
        xlsx_path = f"extracted_form_{ts}.xlsx"

        df.to_csv(csv_path, index=False)
        df.to_excel(xlsx_path, index=False)

        print(f"\n CSV saved: {csv_path}")
        print(f" Excel saved: {xlsx_path}")


# =====================================================
# RUN SCRIPT
# =====================================================
if __name__ == "__main__":
    processor = FormOCRProcessor("TestImage.jpeg")
    df = processor.process()

    print("\n" + "=" * 80)
    print("EXTRACTED DATA (EasyOCR + TrOCR)")
    print("=" * 80)
    print(df)

    processor.export(df)
    print(f"\n Total records extracted: {len(df)}")
