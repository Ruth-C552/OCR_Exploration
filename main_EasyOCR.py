import cv2
import easyocr
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans

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

class KMeansFormOCR:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.reader = easyocr.Reader(["en"], gpu=False)
        self.column_centers = None

    # ---------------- DETECT TEXT ----------------
    def detect_text(self):
        print("Detecting text with EasyOCR...")
        return self.reader.readtext(
            self.gray,
            detail=1,
            paragraph=False
        )

    # ---------------- ROW GROUPING ----------------
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

    # ---------------- KMEANS COLUMNS ----------------
    def detect_columns_kmeans(self, detections):
        x_centers = []

        for bbox, _, _ in detections:
            x_center = int((bbox[0][0] + bbox[1][0]) / 2)
            x_centers.append(x_center)

        x_centers = np.array(x_centers).reshape(-1, 1)

        n_cols = min(len(FORM_COLUMNS), len(x_centers))

        kmeans = KMeans(n_clusters=n_cols, random_state=42, n_init="auto")
        kmeans.fit(x_centers)

        centers = sorted(int(c[0]) for c in kmeans.cluster_centers_)
        self.column_centers = centers

        print("Detected column centers (left → right):")
        for col, cx in zip(FORM_COLUMNS, centers):
            print(f"  {col:20s} @ x ≈ {cx}")

    def assign_column(self, x):
        """
        Assign x-position to nearest KMeans column center
        """
        distances = [abs(x - c) for c in self.column_centers]
        idx = int(np.argmin(distances))
        return FORM_COLUMNS[idx]

    # ---------------- EXTRACT TABLE ----------------
    def extract(self):
        detections = self.detect_text()

        # KMeans column discovery
        self.detect_columns_kmeans(detections)

        rows = self.group_rows(detections)
        records = []

        for r_idx, row in enumerate(rows):
            record = {c: "" for c in FORM_COLUMNS}

            for bbox, text, conf in row["items"]:
                x = int(bbox[0][0])
                y = int(bbox[0][1])
                w = int(bbox[1][0] - bbox[0][0])
                h = int(bbox[2][1] - bbox[0][1])

                if h < 15 or not text.strip():
                    continue

                roi = self.image[y:y+h, x:x+w]
                cv2.imwrite(
                    str(DEBUG_DIR / f"row{r_idx}_{x}_{y}.png"),
                    roi
                )

                col = self.assign_column(x)

                # Append text (important for multi-word fields)
                record[col] += (" " + text.strip())

            if any(v.strip() for v in record.values()):
                records.append(record)

        return pd.DataFrame(records, columns=FORM_COLUMNS)


# ---------------- RUN ----------------
if __name__ == "__main__":
    ocr = KMeansFormOCR(IMAGE_PATH)
    df = ocr.extract()

    print("\n" + "=" * 80)
    print("EXTRACTED DATA (EasyOCR + KMeans)")
    print("=" * 80)
    print(df.fillna(""))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"extracted_{ts}.csv", index=False)
    df.to_excel(f"extracted_{ts}.xlsx", index=False)

    print("\nSaved CSV & Excel")
    print(f"Debug crops saved in → {DEBUG_DIR}")
