from preprocess import preprocess_image
from table_detector import detect_cells
from trocr_ocr import recognize_text
from field_mapper import map_to_fields
import json

IMAGE_PATH = "TestImage.jpeg"

def main():
    original, binary = preprocess_image(IMAGE_PATH)
    cells = detect_cells(binary, original)

    print(f"Detected {len(cells)} cells")

    texts = []
    for _, _, roi in cells:
        text = recognize_text(roi)
        texts.append(text)

    structured_data = map_to_fields(texts)

    with open("output.json", "w") as f:
        json.dump(structured_data, f, indent=4)

    print("Extraction complete. Saved to output.json")

if __name__ == "__main__":
    main()
