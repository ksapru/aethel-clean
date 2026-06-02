import os
import pandas as pd
from pypdf import PdfReader
from typing import List, Dict, Any

class DocumentParser:
    """Parses PE fund documents (PDF, Excel, CSV)."""

    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """Extracts text from PDF files."""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text

    @staticmethod
    def parse_excel(file_path: str) -> str:
        """Extracts data from Excel files and converts to a text representation."""
        xls = pd.ExcelFile(file_path)
        combined_text = ""
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            combined_text += f"Sheet: {sheet_name}\n"
            combined_text += df.to_string(index=False) + "\n\n"
        return combined_text

    @staticmethod
    def parse_document(file_path: str) -> Dict[str, Any]:
        """Auto-detects file type and parses accordingly."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            content = DocumentParser.parse_pdf(file_path)
        elif ext in [".xlsx", ".xls"]:
            content = DocumentParser.parse_excel(file_path)
        elif ext == ".csv":
            content = pd.read_csv(file_path).to_string(index=False)
        elif ext == ".txt":
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return {
            "file_name": os.path.basename(file_path),
            "content": content,
            "metadata": {
                "type": ext,
                "path": file_path
            }
        }
