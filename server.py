from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import fitz
import easyocr
import numpy as np
import cv2
import os
import tempfile
import shutil
import base64

# Initialize the FastAPI App
app = FastAPI(title="Enterprise AI Content & Diagram Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




ocr_reader = easyocr.Reader(['en'], gpu=False)


@app.post("/api/extract")
async def extract_metadata(files: list[UploadFile] = File(...)):
    all_extracted_data = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        try:
            extracted_chunks = []
            extracted_diagrams = []

            # Open with PyMuPDF for advanced analysis (OCR & Diagram Extraction)
            doc = fitz.open(temp_path)

            try:
                # DIAGRAM EXTRACTION LOOP
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    image_list = page.get_images(full=True)

                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Convert raw image bytes to Base64 string for HTML rendering
                        base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
                        data_url = f"data:image/{image_ext};base64,{base64_encoded}"
                        extracted_diagrams.append(data_url)

                # DIGITAL TEXT EXTRACTION LOOP
                with open(temp_path, "rb") as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            extracted_chunks.append(text)

                full_content = "\n\n--- Page Break ---\n\n".join(extracted_chunks)

                # HANDWRITING & OCR FALLBACK LOOP
                if len(full_content.strip()) < 50:
                    ocr_chunks = []
                    for page_num in range(len(doc)):
                        # Render page at 200 DPI for fine handwriting capture
                        pix = doc.load_page(page_num).get_pixmap(dpi=200)
                        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

                        if pix.n == 4:
                            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
                        elif pix.n == 1:
                            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)

                        # Deep Learning Execution with parameters tuned for messy handwriting
                        detection_results = ocr_reader.readtext(
                            img_data,
                            decoder='greedy',
                            paragraph=True,
                            contrast_ths=0.1,
                            adjust_contrast=0.7,
                            slope_ths=0.3
                        )

                        page_ocr_text = "\n".join([line[1] for line in detection_results])

                        if page_ocr_text.strip():
                            ocr_chunks.append(
                                f"[Handwritten / Scanned OCR Page {page_num + 1}]\n" + page_ocr_text.strip())

                    if ocr_chunks:
                        full_content = "\n\n--- Page Break ---\n\n".join(ocr_chunks)

            finally:
                doc.close()

            if not full_content.strip():
                full_content = "[No readable text found on document pages]"

            all_extracted_data.append({
                "file_name": file.filename,
                "status": "Success",
                "full_content": full_content,
                "diagrams": extracted_diagrams
            })

        except Exception as e:
            all_extracted_data.append({
                "file_name": file.filename,
                "status": f"Error: {str(e)}",
                "full_content": "N/A",
                "diagrams": []
            })
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except PermissionError:
                    pass

    return all_extracted_data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)