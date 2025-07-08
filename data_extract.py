# from typing import Dict
# import pdfplumber
# import pytesseract
# from PIL import Image
# from pathlib import Path

# def extract_table_data(file_path: str) -> Dict[str, str]:
#     path = Path(file_path)
#     if path.suffix.lower() == ".pdf":
#         with pdfplumber.open(file_path) as pdf:
#             for page in pdf.pages:
#                 tables = page.extract_tables()
#                 for table in tables:
#                     return table_to_dict(table)
#     else:
#         text = pytesseract.image_to_string(Image.open(file_path))
#         lines = text.splitlines()
#         return extract_table_from_lines(lines)
#     return {}

# def table_to_dict(table):
#     result = {}
#     for row in table:
#         if row and len(row) >= 2:
#             key = row[0].strip()
#             value = ' '.join(cell.strip() for cell in row[1:] if cell)
#             result[key] = value
#     return result

# def extract_table_from_lines(lines):
#     result = {}
#     for line in lines:
#         parts = line.strip().split()
#         if len(parts) >= 2:
#             key = parts[0]
#             value = ' '.join(parts[1:])
#             result[key] = value
#     return result

# # Example usage
# if __name__ == "__main__":
#     file_path = "hba1c.png"
#     table_data = extract_table_data(file_path)
#     for k, v in table_data.items():
#         print(f"{k}: {v}")


#     # glucose = result.get('Glucose')
#     # hba1c = result.get('HBA1C')

#     # print(f"Glucose: {glucose}")
#     # print(f"HbA1c: {hba1c}")
#     # print("\nParsed Key-Value Pairs:")
#     # print(FileProcessingService.parse_key_value_pairs(text))


import os
import base64
import json
from typing import Dict, Optional
import pdfplumber
import pytesseract
from PIL import Image
from pathlib import Path
from openai import OpenAI

class DocumentProcessor:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DocumentProcessor with an optional API key.
        If no API key is provided, it will look for OPENAI_API_KEY in environment variables.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it via constructor or OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def process_document(self, file_path: str) -> Dict:
        """
        Process a document (PDF or image) and extract structured data using GPT-4 Vision.
        
        Args:
            file_path: Path to the PDF or image file
            
        Returns:
            dict: Extracted structured data
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text from the document
        if path.suffix.lower() == ".pdf":
            text = self._extract_text_from_pdf(file_path)
        else:
            text = self._extract_text_from_image(file_path)
        
        # Get structured data using GPT-4
        return self._get_structured_data(text, str(path))
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using pdfplumber."""
        text = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    
    def _extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image using Tesseract OCR."""
        return pytesseract.image_to_string(Image.open(file_path))
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # def _get_structured_data(self, text: str, file_path: str) -> Dict:
    #     """
    #     Use GPT-4 to extract structured data from text.
        
    #     Args:
    #         text: Extracted text from the document
    #         file_path: Path to the original file (for image processing)
            
    #     Returns:
    #         dict: Extracted structured data
    #     """
    #     # Prepare the prompt
    #     system_prompt = """
    #     You are a medical document processing assistant. Extract the following information from the provided document:
    #     - Patient name
    #     - Date of birth
    #     - Test date
    #     - Test results (HbA1c, Glucose, Cholesterol, etc.) with values and units
    #     - Any other relevant medical information
        
    #     Return the data in a structured JSON format with appropriate fields.
    #     """
        
    #     messages = [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": f"Extract information from this document:\n\n{text}"}
    #     ]
        
    #     # If it's an image, include the image in the request
    #     path = Path(file_path)
    #     if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
    #         base64_image = self._encode_image(file_path)
    #         messages[1]["content"] = [
    #             {"type": "text", "text": "Extract information from this document:"},
    #             {
    #                 "type": "image_url",
    #                 "image_url": f"data:image/{path.suffix[1:]};base64,{base64_image}"
    #             }
    #         ]
        
    #     try:
    #         response = self.client.chat.completions.create(
    #             model="gpt-4" if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] else "gpt-4",
    #             messages=messages,
    #             max_tokens=1000,
    #             response_format={"type": "json_object"}
    #         )
            
    #         # Parse the response
    #         result = response.choices[0].message.content
    #         return json.loads(result)
            
    #     except Exception as e:
    #         return {"error": str(e), "extracted_text": text}

    def _get_structured_data(self, text: str, file_path: str) -> Dict:
        """
        Use GPT-4 to extract structured data from text.
    
        Args:
            text: Extracted text from the document
            file_path: Path to the original file (for image processing)
        
        Returns:
            dict: Extracted structured data
        """
        # Prepare the prompt
        system_prompt = """
        You are a medical document processing assistant. Extract the following information from the provided document:
        - Patient name
        - Date of birth
        - Test date
        - Test results (HbA1c, Glucose, Cholesterol, etc.) with values and units
        - Any other relevant medical information
    
        Return the data in a structured JSON format with appropriate fields.
    """
    
        messages = [
            {"role": "system", "content": system_prompt}
        ]
    
        # If it's an image, include the image in the request
        path = Path(file_path)
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            base64_image = self._encode_image(file_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract information from this document:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{path.suffix[1:]};base64,{base64_image}"
                        }
                    }
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Extract information from this document:\n\n{text}"
            })
    
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o" if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] else "gpt-4",
                messages=messages,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
        
            # Parse the response
            result = response.choices[0].message.content
            return json.loads(result)
        
        except Exception as e:
            return {"error": str(e), "extracted_text": text}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract structured data from medical documents using GPT-4')
    parser.add_argument('file_path', type=str, help='Path to the document (PDF or image)')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path (optional)')
    parser.add_argument('--api-key', type=str, help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    
    args = parser.parse_args()
    
    try:
        processor = DocumentProcessor(api_key=args.api_key)
        result = processor.process_document(args.file_path)
        
        # Save or print the result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()