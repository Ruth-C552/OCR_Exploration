import cv2
import numpy as np
import easyocr
import pandas as pd
from typing import List, Dict, Tuple
import re


class FormOCRProcessor:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.image = None
        self.processed_image = None

    # -------------------------------------------------
    # IMAGE PREPROCESSING
    # -------------------------------------------------
    def load_and_preprocess_image(self) -> np.ndarray:
        self.image = cv2.imread(self.image_path)

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 10
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        self.processed_image = processed
        return processed

    # -------------------------------------------------
    # OCR
    # -------------------------------------------------
    def extract_text_with_positions(self) -> List[Tuple]:
        return self.reader.readtext(self.processed_image)

    # -------------------------------------------------
    # ROW GROUPING (CRITICAL FIX)
    # -------------------------------------------------
    def group_by_rows(self, ocr_results, row_threshold=25):
        rows = []

        for bbox, text, conf in sorted(ocr_results, key=lambda x: x[0][0][1]):
            y = bbox[0][1]
            placed = False

            for row in rows:
                if abs(row["y"] - y) < row_threshold:
                    row["items"].append((bbox, text))
                    placed = True
                    break

            if not placed:
                rows.append({"y": y, "items": [(bbox, text)]})

        return rows

    # -------------------------------------------------
    # COLUMN MAPPING (BASED ON YOUR FORM)
    # -------------------------------------------------
    def get_column(self, x):
        if x < 120:
            return "HTS Number"
        elif x < 260:
            return "Date of Visit"
        elif x < 460:
            return "Name"
        elif x < 580:
            return "Contact"
        elif x < 650:
            return "Age"
        elif x < 720:
            return "Sex"
        elif x < 820:
            return "Children Alive"
        elif x < 960:
            return "Marital Status"
        elif x < 1080:
            return "Tested Before"
        else:
            return "Test Result"

    # -------------------------------------------------
    # PARSE FORM DATA (REPLACED LOGIC)
    # -------------------------------------------------
    def parse_form_data(self, ocr_results):
        rows = self.group_by_rows(ocr_results)
        records = []

        for row in rows:
            record = {}

            for bbox, text in row["items"]:
                text = text.strip()
                if not text:
                    continue

                x = bbox[0][0]
                column = self.get_column(x)

                # Skip headers
                if text.upper() in ["NAME", "AGE", "SEX"]:
                    continue

                text = re.sub(r"[|_]", "", text)

                record[column] = text

            if "HTS Number" in record and record["HTS Number"].isdigit():
                records.append(record)

        return records

    # -------------------------------------------------
    # CLEAN DATA
    # -------------------------------------------------
    def clean_and_validate_data(self, records):
        cleaned = []

        for r in records:
            row = {}

            for key, value in r.items():
                value = value.strip()

                if key == "Age":
                    m = re.search(r"\d+", value)
                    if m:
                        row[key] = int(m.group())
                elif key == "Sex":
                    row[key] = value.upper()[0]
                else:
                    row[key] = value

            cleaned.append(row)

        return cleaned

    # -------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------
    def visualize_detections(self, ocr_results, output_path="annotated_form.jpg"):
        annotated = self.image.copy()

        for bbox, text, conf in ocr_results:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
            cv2.putText(
                annotated, text,
                (int(bbox[0][0]), int(bbox[0][1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1
            )

        cv2.imwrite(output_path, annotated)
        print(f"Annotated image saved to {output_path}")

    # -------------------------------------------------
    # MAIN PIPELINE
    # -------------------------------------------------
    def process_form(self, visualize=True):
        print("Loading and preprocessing image...")
        self.load_and_preprocess_image()

        print("Extracting text...")
        ocr_results = self.extract_text_with_positions()
        print(f"Found {len(ocr_results)} text elements")

        if visualize:
            self.visualize_detections(ocr_results)

        print("Parsing form data...")
        parsed = self.parse_form_data(ocr_results)

        print("Cleaning data...")
        cleaned = self.clean_and_validate_data(parsed)

        return pd.DataFrame(cleaned)

    # -------------------------------------------------
    # EXPORT
    # -------------------------------------------------
    def export_to_csv(self, df, path="extracted_form_data.csv"):
        df.to_csv(path, index=False)
        print(f"CSV saved: {path}")

    def export_to_excel(self, df, path="extracted_form_data.xlsx"):
        df.to_excel(path, index=False)
        print(f"Excel saved: {path}")


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    processor = FormOCRProcessor("TestImage.jpeg")

    df = processor.process_form(visualize=True)

    print("\n" + "=" * 80)
    print("EXTRACTED DATA")
    print("=" * 80)
    print(df)

    processor.export_to_csv(df)
    processor.export_to_excel(df)

    print("\nTotal records extracted:", len(df))
