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

    #---------------- Deskew correction ----------------
    def deskew(self, img):
        """
        Deskew image using minimum-area rectangle angle
        Works very well for slanted handwritten digits
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)

        thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return img  # too small to estimate angle

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    # ---------------- TrOCR ----------------
    def recognize_with_trocr(self, roi):
        pil_img = Image.fromarray(roi).convert("RGB")
        pixel_values = self.processor(images=pil_img, return_tensors="pt") \
            .pixel_values.to(self.device)

        with torch.no_grad():
            ids = self.model.generate(pixel_values, max_new_tokens=16)

        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    #----------------- Digit Specific OCR ----------------
    def preprocess_numeric_roi(self, roi):
        roi = self.deskew(roi)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21, 12
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        return th

    # ---------------- NUMERIC OCR ----------------
    def recognize_numeric(self, roi):
        roi = self.preprocess_numeric_roi(roi)

        results = self.detector.readtext(
            roi,
            allowlist="0123456789/.-",
            text_threshold=0.3,
            low_text=0.2,
            detail=0
        )

        if not results:
            return ""

        text = "".join(results)

        # Fix handwritten confusions
        text = text.replace("O", "0").replace("o", "0")
        text = text.replace("I", "1").replace("l", "1")
        text = text.replace("S", "5").replace("Z", "2")

        text = re.sub(r"[^\d/.\-]", "", text)
        return text.strip()

    # ---------------- ROW GROUPING ----------------
    def group_rows(self, detections, threshold=40):
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

        if record["Children Alive"]:
            m = re.search(r"\d+", record["Children Alive"])
            record["Children Alive"] = int(m.group()) if m else None

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

    def is_valid_numeric(self, text):
        """
        Accept only clean numeric strings for numeric columns
        """
        if not text:
            return False

        return bool(re.fullmatch(r"[0-9/.\-]+", text))

    # ---------------- TABLE EXTRACTION ----------------
    def extract_table(self, detections):
        rows = self.group_rows(detections)
        records = []

        NUMERIC_COLS = {
            "HTS Number",
            "Date of Visit",
            "Age",
            "Children Alive",
            "Contact"
        }

        for row in rows:
            record = {col: "" for col in FORM_COLUMNS}

            for bbox, text, conf in row["items"]:
                x = int(bbox[0][0])
                y = int(bbox[0][1])
                w = int(bbox[1][0] - bbox[0][0])
                h = int(bbox[2][1] - bbox[0][1])

                if h < 28:
                    continue

                roi = self.image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                col = self.assign_column(x)

                if col in NUMERIC_COLS and w > 120:
                    final_text = self.recognize_numeric(roi)
                    if not final_text or not self.is_valid_numeric(final_text):
                        continue
                elif conf < 0.65:
                    final_text = self.recognize_with_trocr(
                        cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    )
                else:
                    final_text = text

                final_text = re.sub(r"\s+", " ", final_text).strip()
                if final_text:
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
    processor = FormOCRProcessor("TestImage2.jpeg")
    df = processor.process()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.fillna(""))

    processor.export(df)
