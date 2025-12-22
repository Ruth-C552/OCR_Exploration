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

#--------- Auto detect column --------------------
    def detect_column_boundaries(self, detections):
        """
        Automatically infer column X-centers from detected text boxes
        """
        x_centers = []

        for bbox, _, _ in detections:
            x_center = int((bbox[0][0] + bbox[1][0]) / 2)
            x_centers.append(x_center)

        x_centers = np.array(x_centers).reshape(-1, 1)

        # Number of columns on the form
        n_columns = min(len(FORM_COLUMNS), len(x_centers))

        kmeans = KMeans(n_clusters=n_columns, random_state=42)
        kmeans.fit(x_centers)

        centers = sorted(int(c[0]) for c in kmeans.cluster_centers_)
        return centers

#--------- MAP X POSITION TO FORM COLUMN ----------
    def column_from_x(self, x):
        for idx, center in enumerate(self.column_centers):
            if x < center:
                col = FORM_COLUMNS[idx]
                if col in ["Name", "Residential Address"]:
                    return "NAME_ADDRESS"
                return col
        return FORM_COLUMNS[-1]


#----------- EXTRACT TABLE DATA --------------
    def extract_table(self, detections):
        rows = self.group_rows(detections)
        records = []

        print(f" Processing {len(rows)} detected rows...\n")

        for i, row in enumerate(rows):
            print(f" Row {i + 1}/{len(rows)}")

            # Initialize FULL row with empty values
            record = {col: None for col in FORM_COLUMNS}
            
            name_candidate = None
            address_candidate = None

            for bbox in row["boxes"]:
                x = int(bbox[0][0])
                y = int(bbox[0][1])
                w = int(bbox[1][0] - bbox[0][0])
                h = int(bbox[2][1] - bbox[0][1])
                
                if h < 25:
                    continue

                roi = self.image[y:y+h, x:x+w]
                if roi.size == 0: 
                    continue

                column = self.column_from_x(x)
                print(f"   TrOCR → {column}")

                text = self.recognize_with_trocr(roi)
                text = re.sub(r"[^A-Za-z0-9+/.\- ]", "", text).strip()

                if not text:
                    continue
                
                #Split name vs Address
                if column == "NAME_ADDRESS":
                    if len(text.split()) <= 4:
                        name_candidate = text
                    else:
                        address_candidate = text
                else:
                    record[column] = text
                    
            #assign split values
            if name_candidate:
                record["Name"] = name_candidate
            if address_candidate:
                record["Residential Address"] = address_candidate

            # Accept row if it contains ANY meaningful data
            if any(record.values()):
                records.append(record)

        return records


#------------- MAIN PIPELINE ----------------
    def process(self):
        print("\n Starting OCR pipeline...\n")
        self.preprocess()
        detections = self.detect_regions()
        
        self.column_centers = self.detect_column_boundaries(detections)
        print("Detected column centers:", self.column_centers)
        
        self.visualize_detections(detections) #visualization 
        
        records = self.extract_table(detections)

        # Enforce consistent column order
        df = pd.DataFrame(records, columns=FORM_COLUMNS)
        return df

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
