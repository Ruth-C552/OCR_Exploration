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

NUMERIC_COLS = {"HTS Number", "Date of Visit", "Age", "Children Alive", "Contact"}


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

        # Slight blur + adaptive threshold
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
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
        for bbox, text, _ in detections:
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

    # ---------------- NUMERIC OCR ----------------
    def fix_numeric_errors(self, text):
        mapping = str.maketrans({"O":"0", "I":"1", "l":"1", "S":"5", "B":"8"})
        text = text.translate(mapping)
        return re.sub(r"[^\d/]", "", text)

    def recognize_numeric_smart(self, roi):
        # Deskew ROI first
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray > 0))
        if coords.shape[0] > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_img = cv2.resize(bin_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        results = self.detector.readtext(bin_img, allowlist="0123456789/.-+", detail=0)
        text = "".join(results) if results else ""
        text = self.fix_numeric_errors(text)

        if len(text) < 3:
            try:
                text = self.recognize_with_trocr(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                text = self.fix_numeric_errors(text)
            except:
                pass
        return text.strip()
    
    #---------------- CLEAN NUMERIC BY FIELD ----------------
    def clean_numeric_by_field(self, text, field):
        if not text:
            return ""

        text = text.strip()

        if field == "HTS Number":
            # HTS is usually short digits, sometimes with +
            return re.sub(r"[^\d+]", "", text)

        if field == "Date of Visit":
            # allow dd/mm/yyyy, dd-mm-yyyy, yyyy
            text = text.replace(" ", "")
            m = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})", text)
            return m.group(0) if m else ""

        if field == "Contact":
            # phone numbers
            text = re.sub(r"[^\d+]", "", text)
            return text if len(text) >= 7 else ""

        if field == "Children Alive":
            m = re.search(r"\d+", text)
            return m.group(0) if m else ""

        return text
   

    # ---------------- ROW GROUPING ----------------
    def group_rows(self, detections, threshold=45):
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

        if record["Age"]:
            m = re.search(r"\d{1,3}", record["Age"])
            record["Age"] = int(m.group()) if m else None

        if record["Children Alive"] != "":
            try:
                record["Children Alive"] = int(record["Children Alive"])
            except:
                record["Children Alive"] = None

        if record["Contact"]:
            record["Contact"] = re.sub(r"[^\d+]", "", record["Contact"])

        if record["Date of Visit"]:
            record["Date of Visit"] = record["Date of Visit"].replace(" ", "")

        return record

    # ---------------- VALID ROW ----------------
    def is_valid_data_row(self, record):
        header_keywords = {
            "hts", "date", "name", "address", "contact", "age",
            "sex", "children", "marital", "tested", "result"
        }
        row_text = " ".join(str(v).lower() for v in record.values() if v)
        keyword_hits = sum(1 for k in header_keywords if k in row_text)
        digit_count = sum(c.isdigit() for c in row_text)

        if keyword_hits >= 2 and digit_count == 0:
            return False

        return bool(record["HTS Number"] or record["Name"])

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

                if h < 20:
                    continue

                roi = self.image[max(0, y-2):y+h+2, max(0, x-2):x+w+2]
                if roi.size == 0:
                    continue

                col = self.assign_column(x)

                if col in NUMERIC_COLS:
                    raw_text = self.recognize_numeric_smart(roi)
                    final_text = self.clean_numeric_by_field(raw_text, col)

                    # ---- NUMERIC FALLBACK FOR SHORT / MISSED VALUES ----
                    if col in {"HTS Number", "Children Alive"} and not final_text:
                        fallback = re.sub(r"[^\d]", "", text)
                        if fallback:
                            final_text = fallback


                elif conf < 0.65:
                    final_text = self.recognize_with_trocr(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                else:
                    final_text = text

                final_text = re.sub(r"\s+", " ", final_text).strip()
                if final_text:
                    record[col] = (record.get(col, "") + " " + final_text).strip()

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
    processor = FormOCRProcessor("TestImage2.jpeg")
    df = processor.process()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.fillna(""))

    processor.export(df)
