import cv2
import easyocr
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

IMAGE_PATH = "TestImage2.jpeg"
DEBUG_DIR = Path("debug_cells")
DEBUG_DIR.mkdir(exist_ok=True)

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

# ---- FIXED COLUMN ZONES (adjust once, then stable) ----
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

class SimpleFormOCR:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.reader = easyocr.Reader(["en"], gpu=False)

    def detect_text(self):
        print("Detecting text with EasyOCR...")
        return self.reader.readtext(
            self.gray,
            detail=1,
            paragraph=False
        )

    def assign_column(self, x):
        for col, (x1, x2) in COLUMN_X_RANGES.items():
            if x1 <= x < x2:
                return col
        return None

    def group_rows(self, detections, y_thresh=20):
        rows = []

        for det in sorted(detections, key=lambda d: d[0][0][1]):
            bbox, text, conf = det
            y = bbox[0][1]

            placed = False
            for row in rows:
                if abs(row["y"] - y) < y_thresh:
                    row["items"].append(det)
                    placed = True
                    break

            if not placed:
                rows.append({"y": y, "items": [det]})

        return rows

    def extract(self):
        detections = self.detect_text()
        rows = self.group_rows(detections)

        records = []

        for r_idx, row in enumerate(rows):
            record = {c: "" for c in FORM_COLUMNS}

            for bbox, text, conf in row["items"]:
                x = int(bbox[0][0])
                y = int(bbox[0][1])
                w = int(bbox[1][0] - bbox[0][0])
                h = int(bbox[2][1] - bbox[0][1])

                if h < 15 or len(text.strip()) == 0:
                    continue

                roi = self.image[y:y+h, x:x+w]
                cv2.imwrite(
                    str(DEBUG_DIR / f"row{r_idx}_{x}_{y}.png"),
                    roi
                )

                col = self.assign_column(x)
                if not col:
                    continue

                # Append text (important for Name & Address)
                record[col] += (" " + text.strip())

            if any(v.strip() for v in record.values()):
                records.append(record)

        return pd.DataFrame(records, columns=FORM_COLUMNS)


# ----------------- RUN -----------------
if __name__ == "__main__":
    ocr = SimpleFormOCR(IMAGE_PATH)
    df = ocr.extract()

    print("\n" + "=" * 80)
    print("EXTRACTED DATA")
    print("=" * 80)
    print(df.fillna(""))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"extracted_{ts}.csv", index=False)
    df.to_excel(f"extracted_{ts}.xlsx", index=False)

    print("\nSaved CSV & Excel")
    print(f"Debug crops saved in → {DEBUG_DIR}")
