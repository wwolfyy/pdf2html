from typing import Dict
from fastapi import FastAPI, File, UploadFile
from convert_pdf import convert_pdf

app = FastAPI()

@app.post("/convert_pdf")
async def convert_pdf_endpoint(file: UploadFile) -> Dict:
    pdf_bytes = await file.read()
    result = convert_pdf(pdf_bytes)
    return result