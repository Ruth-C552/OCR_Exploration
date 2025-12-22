import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import os
from pathlib import Path

class HandwrittenFormOCR:
    def __init__(self, model_name="microsoft/trocr-large-handwritten"):
        """Initialize TrOCR model and processor"""
        print("Loading TrOCR model...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess the image"""
        # Check if file exists
        if not os.path.exists(image_path):
            # Try to find the file in the current directory
            current_dir = os.getcwd()
            print(f"Current directory: {current_dir}")
            print(f"Files in current directory: {os.listdir(current_dir)}")
            
            # Look for image files
            image_files = [f for f in os.listdir(current_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            if image_files:
                print(f"\nFound image files: {image_files}")
                print(f"Please use one of these filenames or provide the full path.")
            
            raise ValueError(f"Could not find image at: {image_path}\nPlease check the file path and name.")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path} - file may be corrupted")
        
        print(f"Image loaded successfully: {image_path}")
        print(f"Image shape: {img.shape}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deskew the image
        gray = self.deskew_image(gray)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew the image to correct rotation"""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Only deskew if angle is significant
        if abs(angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            return rotated
        return image
    
    def detect_table_cells(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect table cell boundaries"""
        # Apply binary threshold
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_structure = cv2.add(horizontal_lines, vertical_lines)
        
        # Find contours
        contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort cells
        cells = []
        min_cell_area = 1000  # Minimum area to be considered a cell
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > min_cell_area and w > 30 and h > 20:
                cells.append((x, y, w, h))
        
        # Sort cells by position (top to bottom, left to right)
        cells.sort(key=lambda c: (c[1], c[0]))
        
        return cells
    
    def extract_text_from_cell(self, image: np.ndarray, cell: Tuple[int, int, int, int]) -> str:
        """Extract text from a single cell using TrOCR"""
        x, y, w, h = cell
        
        # Add padding
        padding = 5
        x = max(0, x + padding)
        y = max(0, y + padding)
        w = max(1, w - 2*padding)
        h = max(1, h - 2*padding)
        
        # Crop cell
        cell_img = image[y:y+h, x:x+w]
        
        # Skip if cell is too small or empty
        if cell_img.size == 0 or w < 10 or h < 10:
            return ""
        
        # Check if cell is mostly empty (white)
        if np.mean(cell_img) > 250:
            return ""
        
        # Convert to PIL Image
        pil_img = Image.fromarray(cell_img).convert("RGB")
        
        # Process with TrOCR
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text.strip()
    
    def organize_cells_into_rows(self, cells: List[Tuple[int, int, int, int]], 
                                 tolerance: int = 20) -> List[List[Tuple[int, int, int, int]]]:
        """Organize cells into rows based on y-coordinate"""
        if not cells:
            return []
        
        rows = []
        current_row = [cells[0]]
        current_y = cells[0][1]
        
        for cell in cells[1:]:
            if abs(cell[1] - current_y) < tolerance:
                current_row.append(cell)
            else:
                # Sort current row by x-coordinate
                current_row.sort(key=lambda c: c[0])
                rows.append(current_row)
                current_row = [cell]
                current_y = cell[1]
        
        # Add last row
        if current_row:
            current_row.sort(key=lambda c: c[0])
            rows.append(current_row)
        
        return rows
    
    def process_form(self, image_path: str, visualize: bool = True) -> pd.DataFrame:
        """Process the entire form and extract data"""
        print("Preprocessing image...")
        preprocessed = self.preprocess_image(image_path)
        
        print("Detecting table cells...")
        cells = self.detect_table_cells(preprocessed)
        print(f"Found {len(cells)} cells")
        
        print("Organizing cells into rows...")
        rows = self.organize_cells_into_rows(cells)
        print(f"Organized into {len(rows)} rows")
        
        # Visualize detected cells if requested
        if visualize:
            self.visualize_cells(cv2.imread(image_path), cells)
        
        print("Extracting text from cells...")
        data = []
        for row_idx, row_cells in enumerate(rows):
            row_data = []
            for cell_idx, cell in enumerate(row_cells):
                text = self.extract_text_from_cell(preprocessed, cell)
                row_data.append(text)
                print(f"Row {row_idx+1}, Cell {cell_idx+1}: {text}")
            data.append(row_data)
        
        # Create DataFrame with proper columns based on the form structure
        columns = ['HTS Number', 'Date of Visit', 'Name/Address', 'Contact Details', 
                   'Age', 'Sex', 'Children Alive', 'Marital Status', 'Tested Before', 'Test Result']
        
        # Pad rows to match column count
        max_cols = max(len(row) for row in data) if data else len(columns)
        for row in data:
            while len(row) < max_cols:
                row.append("")
        
        df = pd.DataFrame(data)
        
        # Set column names (adjust based on actual structure)
        if len(df.columns) <= len(columns):
            df.columns = columns[:len(df.columns)]
        
        return df
    
    def visualize_cells(self, image: np.ndarray, cells: List[Tuple[int, int, int, int]]):
        """Visualize detected cells on the image"""
        vis_image = image.copy()
        for cell in cells:
            x, y, w, h = cell
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Table Cells")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    # Initialize the OCR system
    ocr = HandwrittenFormOCR()
    
    # Get the image path - update this to match your actual file
    # Option 1: Use a specific filename
    image_path = "TestImage2.jpeg"
    
    # Option 2: Or let the user input the path
    # image_path = input("Enter the path to your form image: ").strip()
    
    # Option 3: Or automatically find image in current directory
    # current_dir = os.getcwd()
    # image_files = [f for f in os.listdir(current_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # if image_files:
    #     print(f"Found images: {image_files}")
    #     image_path = image_files[0]
    #     print(f"Using: {image_path}")
    
    print(f"Looking for image: {image_path}")
    print(f"In directory: {os.getcwd()}")
    
    try:
        results_df = ocr.process_form(image_path, visualize=True)
        
        # Display results
        print("\n" + "="*80)
        print("EXTRACTED DATA:")
        print("="*80)
        print(results_df.to_string())
        
        # Save to CSV
        output_path = "extracted_form_data.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nData saved to {output_path}")
        
    except Exception as e:
        print(f"\nError processing form: {e}")
        import traceback
        traceback.print_exc()