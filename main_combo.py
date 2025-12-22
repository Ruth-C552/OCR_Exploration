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

COLUMN_X_RANGES = {
    "HTS Number": (0, 120),
    "Date of Visit": (120, 220),
    "Name": (220, 360),
    "Residential Address": (360, 520),
    "Contact": (520, 640),
    "Age": (640, 700),
    "Sex": (700, 760),
    "Children Alive": (760, 850),
    "Marital Status": (850, 950),
    "Tested Before": (950, 1050),
    "Test Result": (1050, 1200),
}

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


 # ---------------- PREPROCESS IMAGE ------------
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

#---------------- EASYOCR: DETECT TEXT REGIONS ------------------
    def detect_regions(self):
        print(" Detecting text regions with EasyOCR...")
        detections = self.detector.readtext(self.processed)
        print(f" Detected {len(detections)} text regions\n")
        return detections

#----------- Assign COLUMN based on X coordinate --------------
    def assign_column(self, x):
        for col, (x1, x2) in COLUMN_X_RANGES.items():
            if x1 <= x < x2:
                return col
        return None

# --------------- TrOCR: HANDWRITING RECOGNITION ----------------
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


#----------- GROUP DETECTIONS INTO ROWS --------------
    def group_rows(self, detections, threshold=18):
        rows = []

        for det in sorted(detections, key=lambda x: x[0][0][1]):
            bbox, text, conf = det
            y = bbox[0][1]
            placed = False

            for row in rows:
                if abs(row["y"] - y) < threshold:
                    row["items"].append(det)
                    placed = True
                    break

            if not placed:
                rows.append({"y": y, "items": [det]})

        return rows

#----------- EXTRACT TABLE DATA --------------
    def extract_table(self, detections):
        rows = self.group_rows(detections)
        records = []

        print(f" Processing {len(rows)} detected rows...\n")

        for i, row in enumerate(rows):
            print(f" Row {i + 1}/{len(rows)}")

            # Initialize FULL row with empty values
            record = {col: "" for col in FORM_COLUMNS}

            for bbox,text, conf in row["items"]:
                x = int(bbox[0][0])
                y = int(bbox[0][1])
                w = int(bbox[1][0] - bbox[0][0])
                h = int(bbox[2][1] - bbox[0][1])
                
                if h < 18:
                    continue

                roi = self.image[y:y+h, x:x+w]
                if roi.size == 0: 
                    continue

                 #Hybrid OCR approach
                '''if conf < 0.60:
                    final_text = self.recognize_with_trocr(roi)
                else:
                    final_text = text'''
                    
                # Improved hybrid OCR decision
                use_trocr = (conf < 0.45) or (h > 30)
                
                final_text = (
                    self.recognize_with_trocr(roi) if use_trocr else text
                )                   
                    
                # clean text    
                final_text = re.sub(r"[^A-Za-z0-9+/.\- ]", "", final_text).strip()
                if not final_text:
                    continue
                
                col = self.assign_column(x)
                if not col:
                    continue
                
                record[col] += " " + final_text
 
            #Clean whitespace
            record = {k: v.strip() for k, v in record.items()}
            
            if re.fullmatch(r"h\s*t\s*s(\s*number)?", record["HTS Number"].lower()):
              continue

            # Accept row if it contains ANY meaningful data
            if any(record.values()):
                records.append(self.normalize_record(record))

        return records


#------------- MAIN PIPELINE ----------------
    def process(self):
        print("\n Starting OCR pipeline...\n")
        self.preprocess()
        detections = self.detect_regions()
        
#        self.column_centers = self.detect_column_boundaries(detections)
#        print("Detected column centers:", self.column_centers)
        
        self.visualize_detections(detections) #visualization 
        
        records = self.extract_table(detections)

        # Enforce consistent column order
        df = pd.DataFrame(records, columns=FORM_COLUMNS)
        return df

#------------- Post-processing text  ----------------
    def normalize_record(self, record):
        if record["Sex"]:
            record["Sex"] = record["Sex"].strip().upper()[0]

        if record["Age"]:
            m = re.search(r"\d+", record["Age"])
            record["Age"] = int(m.group()) if m else None

        if record["Children Alive"]:
            m = re.search(r"\d+", record["Children Alive"])
            record["Children Alive"] = int(m.group()) if m else None

        return record

#------------ EXPORT RESULTS ------------------
    def export(self, df):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"extracted_form_{ts}.csv"
        xlsx_path = f"extracted_form_{ts}.xlsx"

        df.to_csv(csv_path, index=False)
        df.to_excel(xlsx_path, index=False)

        print(f"\n CSV saved: {csv_path}")
        print(f" Excel saved: {xlsx_path}")

#Visualize bbox + detected text for debugging
    def visualize_detections(self, detections, output_path="annotated_detections.jpg"):
        annotated = self.image.copy()

        for bbox, text, conf in detections:
            pts = np.array(bbox, dtype=np.int32)

            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)

            x, y = int(bbox[0][0]), int(bbox[0][1])
            label = f"{text}"

            cv2.putText(
                annotated,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(output_path, annotated)
        print(f" Annotated image saved → {output_path}")


#-------------- RUN SCRIPT --------------------
if __name__ == "__main__":
    processor = FormOCRProcessor("TestImage2.jpeg")
    df = processor.process()

    print("\n" + "=" * 80)
    print("EXTRACTED DATA (EasyOCR + TrOCR)")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.fillna(""))

    processor.export(df)
    print(f"\n Total records extracted: {len(df)}")
