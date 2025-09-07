#!/usr/bin/env python3

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from app.diff import PDFDiffer
from app.analysis import DiffAnalyzer

app = FastAPI(title="Legal Document Change Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/pdf-diff")
async def analyze_pdf_diff(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    if not file1.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="file1 must be a PDF")
    if not file2.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="file2 must be a PDF")
    
    try:
        pdf1_bytes = await file1.read()
        pdf2_bytes = await file2.read()
        
        differ = PDFDiffer()
        result = differ.compare(pdf1_bytes, pdf2_bytes)
        
        analyzer = DiffAnalyzer(result.changes)
        analyzer.analyze_changes()
        
        return JSONResponse(content=analyzer.to_dict())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Legal Document Change Analyzer API", "endpoint": "/pdf-diff"}