# %%
import os
import fitz
from fontTools.ttLib import TTFont

# %% get font data

def get_font_names(pdf_path):
    doc = fitz.open(pdf_path)
    fonts = set()
    
    for page in doc:
        page_fonts = page.get_fonts(full=True)
        for font in page_fonts:
            fonts.add(font[3])
    
    return fonts


def extract_fonts(pdf_path, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    doc = fitz.open(pdf_path)
    font_count = 0

    for page in doc:
        page_fonts = page.get_fonts(full=True)
        for font in page_fonts:
            xref = font[-1]
            if xref is not None:
                font_filename = os.path.join(output_directory, f"font_{font_count}.bin")
                font_data = doc.xref_object(xref)
                with open(font_filename, "wb") as font_file:
                    font_file.write(font_data.encode("latin1"))
                font_count += 1
                print(f"Extracted: {font_filename}")


# %%
def get_text_line_fonts(pdf_path):
    doc = fitz.open(pdf_path)
    result = {}

    font_info_keys = ["xref", "ext", "type", "basefont", "name", "encoding", "object"]

    for page_number, page in enumerate(doc):
        page_result = {}
        blocks = page.get_text("dict", flags=11)["blocks"]
        line_number = 1

        # Get the fonts from the page
        page_fonts = page.get_fonts(full=True)        

        for block in blocks:
            if block["type"] == 0:  # Ignore image blocks
                for line in block["lines"]:
                    font_info = {}
                    for span in line["spans"]:
                        
                        font_name = span["font"]
                        basefonts = [fn for fn in page_fonts if font_name in fn[3]]
                        assert(len(basefonts) >= 1)
                        if len(basefonts) > 1:
                            print(f"Page {page_number}, line {line_number}: Multiple fonts found for {font_name}: using the first one")
                            basefont_instances = len(basefonts)
                        else: 
                            basefont_instances = 1
                        font_details = basefonts[0]                        
                        # font_details = next((f for f in page_fonts if font_name in f[3]), None)
                        if font_details:
                            font_els = font_info_keys[:len(font_details)]
                            font_info[font_name] = dict(zip(font_els, font_details))
                            # font_info[font_name] = {
                            #     "xref": font_details[0],
                            #     "ext": font_details[1],
                            #     "type": font_details[2],
                            #     "basefont": font_details[3],
                            #     "name": font_details[4],
                            #     "encoding": font_details[5],
                            #     "object": font_details[6]
                            # }
                            font_info[font_name].update({'basefont_instances': basefont_instances})
                    page_result[line_number] = font_info
                    line_number += 1
        result[f"Page {page_number + 1}"] = page_result

    return result

# pdf_path = "your_pdf_file.pdf"
text_line_fonts = get_text_line_fonts(pdf_path)

# Example: print the result
for page, page_data in text_line_fonts.items():
    print(page)
    for line, line_data in page_data.items():
        print(f"  Line {line}:")
        for font_name, font_info in line_data.items():
            print(f"    Font name: {font_name}")
            print(f"      Font info: {font_info}")

# %% draw bbox around text lines and display font info using PyMuPDF
def draw_text_line_bboxes(pdf_path, output_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]  # Zero-based index

    for block in page.get_text("dict")["blocks"]:
        if block["type"] == 0:  # Ignore image blocks
            for line in block["lines"]:
                line_text = " ".join([span["text"] for span in line["spans"]]).strip()

                # Skip the line if it contains only whitespace characters
                if not line_text:
                    continue

                # Get the bounding box for the text line
                bbox = fitz.Rect(line["bbox"])
                page.draw_rect(bbox, color=[1, 0, 0], width=1)  # Draw the bounding box

                # Extract and display font names, size, and weight for the line
                font_info = set()
                for span in line["spans"]:
                    font_name = span["font"]
                    font_size = span["size"]
                    font_weight = "bold" if "bold" in font_name.lower() else "normal"
                    font_info.add(f"{font_name} ({font_weight}, {font_size:.1f} pt)")
                font_info_text = f"Fonts: {', '.join(font_info)}"
                page.insert_text((bbox.x0, bbox.y0 - 10), font_info_text, fontsize=8, color=[0, 0, 1])

    single_page_pdf = fitz.open()  # Create a new PDF with a single page
    single_page_pdf.new_page(width=page.rect.width, height=page.rect.height)  # Add a new page with the same dimensions as the original page
    single_page_pdf[0].show_pdf_page(page.rect, doc, pno=page.number)  # Show the modified page on the new single-page PDF
    single_page_pdf.save(output_path)
    single_page_pdf.close()
    doc.close()

pdf_path = "/home/lstm/Github/pdf2html/DATA_sample/보험/adobe/sample_insurance_policy.pdf"
output_pdf_path = "/home/lstm/Github/pdf2html/pymupdf/outputs/lines_with_bboxes.pdf"
page_number = 1
draw_text_line_bboxes(pdf_path, output_pdf_path, page_number)

# %% draw bbox around blocks and display font info using PyMuPDF
def draw_text_block_bboxes(pdf_path, output_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]  # Zero-based index

    shape = page.new_shape()

    # Add a semi-transparent white rectangle over the entire page
    shape.fill_color = [1, 1, 1]
    shape.draw_rect(page.rect)
    # Fill the rectangle with 50% opacity
    shape.finish(fill=True, fill_opacity=0.5)    

    for block in page.get_text("dict")["blocks"]:
        if block["type"] == 0:  # Ignore image blocks
            bbox = fitz.Rect(block["bbox"])  # Get the bounding box for the text block
            
            shape.stroke_color = [1, 0, 0]
            shape.draw_rect(bbox)#, width=1)  # Draw the bounding box
            
            # Extract and display font names, size, and weight for the block
            font_info = set()
            for line in block["lines"]:
                for span in line["spans"]:
                    font_name = span["font"]
                    font_size = span["size"]
                    font_weight = "bold" if "bold" in font_name.lower() else "normal"
                    font_info.add(f"{font_name} ({font_weight}, {font_size:.1f} pt)")
            font_info_text = f"{', '.join(font_info)}"
            shape.insert_text((bbox.x0, bbox.y0 - 3), font_info_text, fontsize=6, color=[0, 0, 1], fontname="Helvetica-Bold")

    shape.commit()

    single_page_pdf = fitz.open()  # Create a new PDF with a single page
    single_page_pdf.new_page(width=page.rect.width, height=page.rect.height)  # Add a new page with the same dimensions as the original page
    single_page_pdf[0].show_pdf_page(page.rect, doc, pno=page.number)  # Show the modified page on the new single-page PDF
    single_page_pdf.save(output_path)
    single_page_pdf.close()
    doc.close()

pdf_path = "/home/lstm/Github/pdf2html/DATA_sample/보험/adobe/sample_insurance_policy.pdf"
output_pdf_path = "/home/lstm/Github/pdf2html/pymupdf/outputs/blocks_with_bboxes.pdf"
page_number = 2
draw_text_block_bboxes(pdf_path, output_pdf_path, page_number)    
