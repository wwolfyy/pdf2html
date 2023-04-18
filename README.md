# pdf2html

# text extraction (Adobe PDF) tools
pypdfium2 (reportedly fast and good)
pymupdf: works, kinda slow

# table extraction (Abode PDF) tools
TBD camelot-py
TBD tabular-py
TBD pdfplumber

# metadat a extraction (Adobe PDF) tools
pymupdf

# layout parsing tools 

## On Adobe format
*OUT* parsr (server-client): take too long, layout/headers not parsed
*OUT* pdfalyzer: decoding problems; complex output
*OUT* PDF Structural Parser: no longer maintained, obscure documentation
TBD Pymupdf (font): 
*OUT* GROBID (server-client): purpose not macthing mine
*PENDING* PDF miner six (bbox & font info): bad documentation; esp bbox coordinates dont' match pymupdf

*OUT* pdfplumber (bbox & font info): limited funtionality; based on pdfminer

## On image
PPStructure: kinda works, may work well with tuning

# ODF-2-html APIs
*OUT* PDFco: way too expensive, can't parse scanned docs