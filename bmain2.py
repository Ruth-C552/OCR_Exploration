import cv2
import numpy as np
import easyocr
import pandas as pd
import re
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from datetime import datetime
from sklearn.cluster import KMeans
from surya import run_ocr


FORM_COLUMNS = [
    "HTS Number",
    "Date of Visit",
    "Name",
    "Residential Address",
    "Contact",
    "Age",
    "Sex",
    "Children Alive",
    "Marital Status",
    "Tested Before",
    "Test Result"
]


class FormOCRProcessor:

    def __init__(self, image_path):
        self.image_path = image_path
        self.detector = easyocr.Reader(['en'], gpu=False)

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
        self.column_centers = {}

    # ---------------- PREPROCESS ----------------
    def preprocess(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError("Image not found")

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 10
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # ---------------- DETECT ----------------
    def detect_regions(self):
        detections = self.detector.readtext(self.processed)
        print(f"Detected {len(detections)} text regions")
        return detections

    # ---------------- COLUMN DETECTION ----------------
    def detect_column_centers(self, detections):
        centers = {}
        for bbox, text, conf in detections:
            t = text.lower()
            x = int(bbox[0][0])
            for col in FORM_COLUMNS:
                if col.lower().split()[0] in t and col not in centers:
                    centers[col] = x
        return centers

    def fallback_column_centers(self, detections):
        xs = np.array([int(bbox[0][0]) for bbox, _, _ in detections]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=len(FORM_COLUMNS), random_state=42)
        kmeans.fit(xs)
        centers = sorted(int(c[0]) for c in kmeans.cluster_centers_)
        return dict(zip(FORM_COLUMNS, centers))

    def assign_column(self, x):
        return min(
            self.column_centers,
            key=lambda c: abs(x - self.column_centers[c])
        )

    # ---------------- TrOCR ----------------
    def recognize_with_trocr(self, roi):
        pil_img = Image.fromarray(roi).convert("RGB")
        pixel_values = self.processor(images=pil_img, return_tensors="pt") \
            .pixel_values.to(self.device)

        with torch.no_grad():
            ids = self.model.generate(pixel_values, max_new_tokens=16)

        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    # ---------------- ROW GROUPING ----------------
    def group_rows(self, detections, threshold=20):
        rows = []
        for det in sorted(detections, key=lambda x: x[0][0][1]):
            y = det[0][0][1]
            for row in rows:
                if abs(row["y"] - y) < threshold:
                    row["items"].append(det)
                    break
            else:
                rows.append({"y": y, "items": [det]})
        print(f"Grouped into {len(rows)} rows")
        return rows

    # ---------------- NORMALIZE ----------------
    def normalize_record(self, record):

        if record["Sex"]:
            record["Sex"] = record["Sex"].strip().upper()[0]

        for col in ("Age", "Children Alive"):
            if record[col]:
                m = re.search(r"\d+", record[col])
                record[col] = int(m.group()) if m else None

        addr = record["Residential Address"].lower()
        if "meanwood" in addr:
            record["Residential Address"] = "Meanwood"

        return record

    # ---------------- VALID ROW ----------------
    def is_valid_data_row(self, record):
        header_keywords = {
            "hts", "number", "date", "visit", "name",
            "address", "contact", "age", "sex", "children",
            "marital", "status", "tested", "before", "result", "test", "alive", "registration", "details",
            "month", "year", "laj", "lcl", "L", "th", "ecexay", "iffrey", "freits", "statu", "resid", "addres", "hitsnumber",
            "laj", "L"
        }
        #combine all text in row
        row_text = " ".join(str(v).lower() for v in record.values() if v)
        
        # reject obvious header rows
        keyword_hits = sum(1 for k in header_keywords if k in row_text)
        digit_count = sum(c.isdigit() for c in row_text)
        if keyword_hits >= 2 and digit_count == 0:
            return False
        
        # require at least one real identifier
        return bool(record["HTS Number"] or record["Name"])

    #----------------- Detect numeric ROI's ----------------
    def is_numeric_heavy(self, text_hint="", roi=None):
        """
        Decide whether ROI should go to SuryaOCR
        """
        if text_hint:
            digits = sum(c.isdigit() for c in text_hint)
            return digits >= max(2, len(text_hint) // 2)

        if roi is not None:
            h, w = roi.shape[:2]
            return w < 160  # numeric cells are usually narrow

        return False

    #-------------------- SuryaOCR Recognition ----------------
    def recognize_with_surya(self, roi):
        pil_img = Image.fromarray(roi).convert("RGB")

        results = run_ocr(
            images=[pil_img],
            langs=["en"],
        )

        if not results or not results[0]:
            return ""

        return " ".join(
            block["text"] for block in results[0] if "text" in block
        ).strip()
    # ---------------- TABLE EXTRACTION ----------------
    def extract_table(self, detections):
        rows = self.group_rows(detections)
        records = []

        for row in rows:
            record = {col: "" for col in FORM_COLUMNS}

            for bbox, text, conf in row["items"]:
                x = int(bbox[0][0])
                y = int(bbox[0][1])
                w = int(bbox[1][0] - bbox[0][0])
                h = int(bbox[2][1] - bbox[0][1])

                if h < 18:
                    continue

                roi = self.image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                # Decide OCR engine
                if self.is_numeric_heavy(text, roi):
                    final_text = self.recognize_with_surya(roi)
                elif conf < 0.65:
                    final_text = self.recognize_with_trocr(roi)
                else:
                    final_text = text

                final_text = re.sub(r"[^A-Za-z0-9+/.\- ]", "", final_text).strip()
                if not final_text:
                    continue

                col = self.assign_column(x)
                record[col] += " " + final_text

            record = {k: v.strip() for k, v in record.items()}

            record = self.normalize_record(record)

            if self.is_valid_data_row(record):
                records.append(record)

        print(f"Extracted {len(records)} valid records")
        return records

    # ---------------- VISUALIZE ----------------
    def visualize_detections(self, detections):
        img = self.image.copy()
        for bbox, _, _ in detections:
            cv2.polylines(img, [np.array(bbox, int)], True, (0, 255, 0), 2)
        cv2.imwrite("annotated_detections.jpg", img)

    # ---------------- MAIN ----------------
    def process(self):
        self.preprocess()
        detections = self.detect_regions()

        self.column_centers = self.detect_column_centers(detections)
        if len(self.column_centers) < len(FORM_COLUMNS) // 2:
            print("Header detection weak — using fallback clustering")
            self.column_centers = self.fallback_column_centers(detections)

        print("Column centers:")
        for k, v in self.column_centers.items():
            print(f"{k:20s} -> x={v}")

        self.visualize_detections(detections)
        return pd.DataFrame(self.extract_table(detections), columns=FORM_COLUMNS)

    # ---------------- EXPORT ----------------
    def export(self, df):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"extracted_{ts}.csv", index=False)
        df.to_excel(f"extracted_{ts}.xlsx", index=False)


# ---------------- RUN ----------------
if __name__ == "__main__":
    processor = FormOCRProcessor("TestImage4.jpeg")
    df = processor.process()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.fillna(""))

    processor.export(df)
