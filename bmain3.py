import cv2
import numpy as np
import easyocr
import pandas as pd
from typing import List, Dict, Tuple
import re
from collections import defaultdict

class FormOCRProcessor:
    def __init__(self, image_path: str):
        """Initialize the OCR processor with an image path."""
        self.image_path = image_path
        self.reader = easyocr.Reader(['en'])
        self.image = None
        self.processed_image = None
        self.extracted_data = []
        
    def load_and_preprocess_image(self) -> np.ndarray:
        """Load and preprocess the image for better OCR results."""
        self.image = cv2.imread(self.image_path)
        
        if self.image is None:
            raise ValueError(f"Could not read image at {self.image_path}")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        self.processed_image = enhanced
        return enhanced
    
    def extract_text_with_positions(self) -> List[Tuple]:
        """Extract text along with bounding box positions."""
        results = self.reader.readtext(self.processed_image)
        return results
    
    def group_text_by_rows(self, ocr_results: List[Tuple], y_tolerance: int = 20) -> List[List[Tuple]]:
        """Group text elements that are on the same horizontal line (row)."""
        if not ocr_results:
            return []
        
        # Sort by y-coordinate
        sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])
        
        rows = []
        current_row = [sorted_results[0]]
        current_y = sorted_results[0][0][0][1]
        
        for result in sorted_results[1:]:
            y_pos = result[0][0][1]
            
            # If within tolerance, add to current row
            if abs(y_pos - current_y) <= y_tolerance:
                current_row.append(result)
            else:
                # Sort current row by x-coordinate (left to right)
                current_row.sort(key=lambda x: x[0][0][0])
                rows.append(current_row)
                current_row = [result]
                current_y = y_pos
        
        # Add last row
        if current_row:
            current_row.sort(key=lambda x: x[0][0][0])
            rows.append(current_row)
        
        return rows
    
    def parse_form_data_improved(self, ocr_results: List[Tuple]) -> List[Dict]:
        """Improved parsing with row-based grouping."""
        print("\n" + "="*80)
        print("RAW OCR RESULTS:")
        print("="*80)
        for i, (bbox, text, conf) in enumerate(ocr_results):
            y_pos = int(bbox[0][1])
            x_pos = int(bbox[0][0])
            print(f"{i:2d}. [{x_pos:4d}, {y_pos:4d}] '{text}' (conf: {conf:.2f})")
        
        # Group text by rows
        rows = self.group_text_by_rows(ocr_results)
        
        print("\n" + "="*80)
        print("GROUPED BY ROWS:")
        print("="*80)
        for i, row in enumerate(rows):
            row_texts = [text for _, text, _ in row]
            print(f"Row {i}: {row_texts}")
        
        # Extract month and year
        month = None
        year = None
        all_text = ' '.join([text for _, text, _ in ocr_results])
        
        if 'DECEMBER' in all_text.upper() or 'DEC' in all_text.upper():
            month = 'December'
        
        year_match = re.search(r'20\d{2}', all_text)
        if year_match:
            year = year_match.group()
        
        # Parse data rows
        records = []
        
        for row in rows:
            row_texts = [text.strip() for _, text, _ in row]
            row_string = ' '.join(row_texts)
            
            # Skip header rows
            if any(header in row_string.upper() for header in 
                   ['MONTH', 'CLIENT', 'REGISTRATION', 'HTS NUMBER', 'DATE OF VISIT', 'NAME']):
                continue
            
            # Look for HTS number pattern (01, 02, 03, 04, etc.)
            hts_match = None
            for text in row_texts:
                if re.match(r'^0\d$', text):
                    hts_match = text
                    break
            
            # If we found an HTS number, this is a data row
            if hts_match:
                record = {
                    'HTS Number': hts_match,
                    'Month': month,
                    'Year': year
                }
                
                # Extract other fields from the row
                for text in row_texts:
                    text_clean = text.strip()
                    
                    # Skip the HTS number itself
                    if text_clean == hts_match:
                        continue
                    
                    # Date pattern
                    if re.search(r'\d+[/-]\d+', text_clean) and 'Date of Visit' not in record:
                        record['Date of Visit'] = text_clean
                    
                    # Age (1-2 digits alone)
                    elif re.match(r'^\d{1,2}$', text_clean) and 'Age' not in record:
                        try:
                            age_val = int(text_clean)
                            if 1 <= age_val <= 120:
                                record['Age'] = age_val
                        except:
                            pass
                    
                    # Sex
                    elif text_clean.upper() in ['M', 'F'] and 'Sex' not in record:
                        record['Sex'] = text_clean.upper()
                    
                    # Marital Status
                    elif any(status in text_clean.upper() for status in 
                            ['MARRIED', 'SINGLE', 'DIVORCED', 'WIDOW', 'MARITAL']):
                        if 'Marital Status' not in record:
                            record['Marital Status'] = text_clean.title()
                    
                    # Test Result
                    elif any(result in text_clean.upper() for result in 
                            ['POSITIVE', 'NEGATIVE', 'POS', 'NEG', 'REAC', 'NON']):
                        if 'Test Result' not in record:
                            record['Test Result'] = text_clean.title()
                    
                    # Name (longer text, letters only, not already captured)
                    elif (len(text_clean) > 2 and 
                          text_clean.replace(' ', '').isalpha() and 
                          'Name' not in record and
                          not any(status in text_clean.upper() for status in 
                                 ['MARRIED', 'SINGLE', 'DIVORCED', 'WIDOW'])):
                        record['Name'] = text_clean.title()
                
                records.append(record)
        
        return records
    
    def visualize_detections(self, ocr_results: List[Tuple], output_path: str = 'annotated_form.jpg'):
        """Visualize detected text on the image."""
        annotated = self.image.copy()
        
        for i, (bbox, text, confidence) in enumerate(ocr_results):
            pts = np.array(bbox, dtype=np.int32)
            
            # Color code by confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.5:
                color = (0, 165, 255)  # Orange - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            cv2.polylines(annotated, [pts], True, color, 2)
            
            # Put text with index
            org = (int(bbox[0][0]), int(bbox[0][1]) - 10)
            cv2.putText(annotated, f"{i}:{text}", org, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 0, 0), 1)
        
        cv2.imwrite(output_path, annotated)
        print(f"\nAnnotated image saved to {output_path}")
        
        return annotated
    
    def process_form(self, visualize: bool = True) -> pd.DataFrame:
        """Main processing pipeline."""
        print("Loading and preprocessing image...")
        self.load_and_preprocess_image()
        
        print("Extracting text...")
        ocr_results = self.extract_text_with_positions()
        
        print(f"Found {len(ocr_results)} text elements")
        
        if visualize:
            self.visualize_detections(ocr_results)
        
        print("\nParsing form data...")
        parsed_data = self.parse_form_data_improved(ocr_results)
        
        # Convert to DataFrame
        df = pd.DataFrame(parsed_data)
        
        return df
    
    def export_to_csv(self, df: pd.DataFrame, output_path: str = 'extracted_form_data.csv'):
        """Export extracted data to CSV."""
        df.to_csv(output_path, index=False)
        print(f"\nData exported to {output_path}")
    
    def export_to_excel(self, df: pd.DataFrame, output_path: str = 'extracted_form_data.xlsx'):
        """Export extracted data to Excel."""
        df.to_excel(output_path, index=False)
        print(f"Data exported to {output_path}")


# Main execution
if __name__ == "__main__":
    import os
    
    # Check if default image exists, otherwise ask
    default_image = 'TestImage2.jpeg'
    
    if os.path.exists(default_image):
        image_path = default_image
        print(f"Using image: {image_path}")
    else:
        image_path = input("Enter the path to your form image: ").strip()
        if not os.path.exists(image_path):
            print(f"Error: File not found at '{image_path}'")
            exit(1)
    
    # Initialize processor
    processor = FormOCRProcessor(image_path)
    
    # Process the form
    df = processor.process_form(visualize=True)
    
    # Display results
    print("\n" + "="*80)
    print("EXTRACTED DATA:")
    print("="*80)
    if len(df) > 0:
        print(df.to_string())
    else:
        print("No data extracted. Check the RAW OCR RESULTS above to see what was detected.")
    print("\n")
    
    # Export results if data was found
    if len(df) > 0:
        processor.export_to_csv(df)
        processor.export_to_excel(df)
        
        print("\n" + "="*80)
        print("STATISTICS:")
        print("="*80)
        print(f"Total records extracted: {len(df)}")
        print(f"Columns: {', '.join(df.columns.tolist())}")
    else:
        print("\n" + "="*80)
        print("No records were extracted.")
        print("Please check the annotated_form.jpg to see what text was detected.")
        print("You may need to:")
        print("  1. Improve image quality (scan at higher resolution)")
        print("  2. Ensure text is clearly visible")
        print("  3. Adjust the preprocessing parameters")
        print("="*80)