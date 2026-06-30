import json
from pypdf import PdfReader
import os
import glob
import fitz  # PyMuPDF
import easyocr
import numpy as np
import cv2


ocr_reader = easyocr.Reader(['en'], gpu=False)


def extract_all_pdfs():
    input_folder = "input_pdfs"
    output_json_path = "extracted_metadata.json"

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created folder '{input_folder}'.")
        return

    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in '{input_folder}'.")
        return

    all_metadata = []

    for pdf_file_path in pdf_files:
        print(f"\nScanning & Processing: {os.path.basename(pdf_file_path)}...")

        try:
            reader = PdfReader(pdf_file_path)
            raw_metadata = reader.metadata
            clean_metadata = {}


            if raw_metadata:
                for key, value in raw_metadata.items():
                    clean_metadata[key.strip('/')] = str(value)

            content_preview = ""
            if len(reader.pages) > 0:
                first_page_text = reader.pages[0].extract_text()
                clean_text = " ".join(first_page_text.split()) if first_page_text else ""

                if len(clean_text) < 20:
                    # Advanced EasyOCR Fallback for scanned documents
                    doc = fitz.open(pdf_file_path)
                    page = doc.load_page(0)

                    # Convert PyMuPDF image to NumPy array for EasyOCR
                    pix = page.get_pixmap(dpi=150)
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

                    if pix.n == 4:
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
                    elif pix.n == 1:
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)


                    detection_results = ocr_reader.readtext(
                        img_data,
                        decoder='greedy',
                        paragraph=True
                    )

                    clean_ocr = " ".join([line[1] for line in detection_results])

                    content_preview = "[EasyOCR Extracted] " + clean_ocr[:250] + "..." if len(
                        clean_ocr) > 10 else "[Image-Only or Scanned PDF with no text]"

                    doc.close()
                else:
                    content_preview = clean_text[:250] + "..." if len(clean_text) > 250 else clean_text
            else:
                content_preview = "[Empty Document]"

            all_metadata.append({
                "file_name": os.path.basename(pdf_file_path),
                "status": "Success",
                "metadata": clean_metadata,
                "preview": content_preview
            })

        except Exception as e:
            all_metadata.append({
                "file_name": os.path.basename(pdf_file_path),
                "status": f"Error: {str(e)}",
                "metadata": {},
                "preview": "N/A"
            })

    # Save the output to JSON
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_metadata, json_file, indent=4)

    print(f"\nSuccess! Processed {len(pdf_files)} files and saved to {output_json_path}.")


if __name__ == "__main__":
    extract_all_pdfs()