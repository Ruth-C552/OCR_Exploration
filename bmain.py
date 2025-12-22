import cv2
import easyocr
import numpy as np
import re
import matplotlib.pyplot as plt
import json

# -----------------------------
# IMAGE PATH
# -----------------------------
image_path = "TestImage2.jpeg"   # adjust if needed

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Image not found")

# -----------------------------
# PREPROCESS
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
processed = cv2.medianBlur(gray, 3)

# -----------------------------
# OCR
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(processed, detail=1)

# -----------------------------
# COLLECT OCR WITH POSITIONS
# -----------------------------
ocr_items = []

for box, text, conf in results:
    if conf < 0.35:
        continue

    text = text.strip()
    if not text:
        continue

    x_center = sum(p[0] for p in box) / 4
    y_center = sum(p[1] for p in box) / 4

    ocr_items.append({
        "text": text,
        "x": x_center,
        "y": y_center
    })

# -----------------------------
# SORT TOP → BOTTOM, LEFT → RIGHT
# -----------------------------
ocr_items.sort(key=lambda i: (i["y"], i["x"]))

# -----------------------------
# GROUP INTO ROWS (BY Y)
# -----------------------------
rows = []
current_row = []
current_y = None
ROW_TOLERANCE = 25

for item in ocr_items:
    if current_y is None or abs(item["y"] - current_y) <= ROW_TOLERANCE:
        current_row.append(item)
        current_y = item["y"]
    else:
        rows.append(current_row)
        current_row = [item]
        current_y = item["y"]

if current_row:
    rows.append(current_row)

# -----------------------------
# EXTRACT RECORDS BY HTS NUMBER
# -----------------------------
records = []
current_record = None

for row in rows:
    row_texts = [r["text"] for r in row]

    # detect HTS number (01, 02, 03, 04)
    hts_match = next(
        (t for t in row_texts if re.fullmatch(r"\d{2}", t)),
        None
    )

    if hts_match:
        if current_record:
            records.append(current_record)

        current_record = {
            "hts_number": hts_match,
            "fields": []
        }

    if current_record:
        for r in row:
            if r["text"] != current_record["hts_number"]:
                current_record["fields"].append(r)

if current_record:
    records.append(current_record)

# -----------------------------
# PRINT CLEAN TERMINAL OUTPUT
# -----------------------------
print("\n===== EXTRACTED RECORDS =====\n")

for rec in records:
    print(f"HTS {rec['hts_number']}:")
    for f in rec["fields"]:
        print(f"  - {f['text']}")
    print()

# Optional JSON output
print(json.dumps(records, indent=2))

# -----------------------------
# VISUAL DEBUGGING (WHAT OCR SEES)
# -----------------------------
debug_img = img.copy()

for box, text, conf in results:
    if conf > 0.35:
        pts = np.array(box, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
        cv2.putText(
            debug_img,
            text,
            (pts[0][0][0], pts[0][0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1
        )

plt.figure(figsize=(10, 14))
plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
