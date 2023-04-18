# %% Calssify lines into headings and paragraphs by font (name, size, color)
from pdfminer.layout import LAParams
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLine, LTChar
from collections import Counter
from pdfminer.pdfcolor import LITERAL_DEVICE_RGB

def classify_line(line, mf_font_name, mf_font_size, mf_font_color):
    font_name, font_size, font_color = line.get_font_info()

    # Modify the classification criteria according to your PDF
    if font_size > mf_font_size or \
        (font_size > mf_font_size and \
         (font_name != mf_font_name or font_color != mf_font_color)):
        return "heading"
    else:
        return "paragraph"

def parse_pdf_layout(pdf_path):
    font_names = []
    font_sizes = []
    font_colors = []
    text_lines = []

    # Update the LAParams to include all_objects
    laparams = LAParams(all_texts=True)
    for page_layout in extract_pages(pdf_path, laparams=laparams):
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                for text_line in element:
                    if isinstance(text_line, LTTextLine):
                        font_info = text_line.get_font_info()
                        if font_info:
                            font_names.append(font_info[0])
                            font_sizes.append(round(font_info[1] + 0.05, 1))
                            font_colors.append(font_info[2])
                        text_lines.append(text_line)

    font_name_counter = Counter(font_names)
    most_frequent_font_name = font_name_counter.most_common(1)[0][0]

    font_size_counter = Counter(font_sizes)
    most_frequent_font_size = font_size_counter.most_common(1)[0][0]

    font_color_counter = Counter(font_colors)
    most_frequent_font_color = font_color_counter.most_common(1)[0][0]

    classified_lines = []
    for line in text_lines:
        classification = classify_line(
            line, 
            most_frequent_font_name,
            most_frequent_font_size, 
            most_frequent_font_color
            )
        classified_lines.append({
             'class': classification,
             'line': line, 
             'font_info': line.get_font_info()
             })

    return classified_lines

def get_font_info(text_line):
    font_name, font_size, font_color = None, None, None

    for char in text_line:
        if isinstance(char, LTChar):
            font_name = char.fontname
            font_size = char.size

            # Get font color information
            if char.ncs and char.graphicstate and char.graphicstate.ncolor:
                font_color = get_color(char.graphicstate.ncolor)
            break

    return font_name, font_size, font_color

def get_color(color_literal):
    color = None
    if color_literal == LITERAL_DEVICE_RGB:
        r, g, b = [c.as_numeric() for c in color_literal.operands]
        color = 'black' if r == 0 and g == 0 and b == 0 else 'other'

    return color

# Add the get_font_info method to the LTTextLine class
LTTextLine.get_font_info = get_font_info

# pdf_path = "/home/lstm/Github/pdf2html/DATA_sample/판례/adobe/특허법원 2017허8459.pdf"
pdf_path = "/home/lstm/Github/pdf2html/DATA_sample/보험/adobe/sample_insurance_policy.pdf"

classified_lines = parse_pdf_layout(pdf_path)

for item in classified_lines:
    print(f"{item['class']}: {item['font_info']}, {item['line']}")

# %%
# Get all headers
headers = [item['line'] for item in classified_lines if item['class'] == 'heading']

for header in headers:
    print(header)

# %%
# %% Classify text blocks into headings and paragraphs by font (name, size, color)
from pdfminer.layout import LAParams
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLine, LTChar
from collections import Counter
from pdfminer.pdfcolor import LITERAL_DEVICE_RGB

def classify_block(block, mf_font_name, mf_font_size, mf_font_color):
    font_name, font_size, font_color = block.get_font_info()

    # Modify the classification criteria according to your PDF
    if font_size > mf_font_size or \
        (font_size > mf_font_size and \
         (font_name != mf_font_name or font_color != mf_font_color)):
        return "heading"
    else:
        return "paragraph"

def parse_pdf_layout(pdf_path):
    font_names = []
    font_sizes = []
    font_colors = []
    text_blocks = []
    page_nums = []

    # Update the LAParams to include all_objects
    laparams = LAParams(all_texts=True)
    # Add a variable to store the current page number
    current_page_number = 0

    for page_layout in extract_pages(pdf_path, laparams=laparams):
        for element in page_layout:            
            if isinstance(element, LTTextBoxHorizontal):
                font_info = element.get_font_info()
                if font_info:
                    font_names.append(font_info[0])
                    font_sizes.append(round(font_info[1] + 0.05, 1))
                    font_colors.append(font_info[2])
                text_blocks.append(element)

                page_nums.append(current_page_number)
        current_page_number += 1     

    font_name_counter = Counter(font_names)
    most_frequent_font_name = font_name_counter.most_common(1)[0][0]

    font_size_counter = Counter(font_sizes)
    most_frequent_font_size = font_size_counter.most_common(1)[0][0]

    font_color_counter = Counter(font_colors)
    most_frequent_font_color = font_color_counter.most_common(1)[0][0]

    classified_blocks = []
    for block, pagenum in zip(text_blocks, page_nums):
        classification = classify_block(
            block, 
            most_frequent_font_name,
            most_frequent_font_size, 
            most_frequent_font_color
            )
        classified_blocks.append({
             'class': classification,
             'block': block, 
             'font_info': block.get_font_info(),
             'page_number': pagenum
             })
        
    return classified_blocks

def get_font_info(text_block):
    font_name, font_size, font_color = None, None, None

    for text_line in text_block:
        if isinstance(text_line, LTTextLine):
            for char in text_line:
                if isinstance(char, LTChar):
                    font_name = char.fontname
                    font_size = char.size

                    # Get font color information
                    if char.ncs and char.graphicstate and char.graphicstate.ncolor:
                        font_color = get_color(char.graphicstate.ncolor)
                    break
            if font_name and font_size and font_color:
                break

    return font_name, font_size, font_color

def get_color(color_literal):
    color = None
    if color_literal == LITERAL_DEVICE_RGB:
        r, g, b = [c.as_numeric() for c in color_literal.operands]
        color = 'black' if r == 0 and g == 0 and b == 0 else 'other'

    return color

# Add the get_font_info method to the LTTextBoxHorizontal class
LTTextBoxHorizontal.get_font_info = get_font_info

# Example PDF paths
# pdf_path = "/home/lstm/Github/pdf2html/DATA_sample/판례/adobe/특허법원 2017허8459.pdf"
pdf_path = "/home/lstm/Github/pdf2html/DATA_sample/보험/adobe/sample_insurance_policy.pdf"

classified_blocks = parse_pdf_layout(pdf_path)

# Print the classified blocks for visual inspection
for block in classified_blocks:
    print(f"{block['class']} : {block['block'].get_text()}")

# %%
# Get all headers
headers = [item['block'] for item in classified_blocks if item['class'] == 'heading']

for header in headers:
    print(header)    


# %% draw bounding box around each block, using "classified_blocks" variable
import fitz  # PyMuPDF

def draw_bbox_and_display_font_info(pdf_path, classified_blocks, output_pdf_path):
    # Open the PDF document
    doc = fitz.open(pdf_path)

    # Define a color for the bounding boxes
    bbox_color = {'stroke': fitz.utils.getColor('red'), 'fill': None}

    for block in classified_blocks:
        # Get block information
        block_obj = block['block']
        bbox = block_obj.bbox
        font_info = block['font_info']

        # Define font properties for font information text
        font_info_text = f"Font: {font_info[0]}, Size: {font_info[1]}, Color: {font_info[2]}"
        font_props = {
            'fontname': 'Helvetica',
            'fontsize': 8,
            'color': fitz.utils.getColor('blue')
        }

        # Get the corresponding page and draw a rectangle around the block
        print(block['page_number'])
        page = doc[block['page_number']]
        page.draw_rect(bbox, color=bbox_color['stroke'], fill=(1,1,1), fill_opacity=0.5, overlay=True)

        # Display the font information just above the bounding box
        page.insert_text((bbox[0], bbox[1] - 10), font_info_text, **font_props)

    # Save the modified PDF to a new file
    doc.save(output_pdf_path)
    doc.close()

output_pdf_path = "/home/lstm/Github/pdf2html/pymupdf/outputs/sample_insurance_policy_with_bbox_and_font_info.pdf"
draw_bbox_and_display_font_info(pdf_path, classified_blocks, output_pdf_path)


# %% draw bounding box and display font info on blocks, using reportlab
