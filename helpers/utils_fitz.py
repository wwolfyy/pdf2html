import fitz
from PIL import Image
from io import BytesIO
from operator import itemgetter
import numpy as np
import pandas as pd
import json

def get_text_length(page):
    
    return len(page.get_text().strip())


def get_num_images(page):        
    
    image_list = page.get_images(full=True)

    return(len(image_list))


def is_scanned_pdf(
        doc: fitz.Document,
        max_text_length: int=100,
        std_pix_threhold: float=5, 
        display_page_bbox: bool=False,
        display_image: bool=False,
        # display_stats: bool=False,
        return_stats: bool=False,
        try_first_n_pages: int=3
        ):
    """
    Check whether a document is scanned image in PDF format
    1. check if the largest image is blank (if blank, it's likely background --> suggests non-scanned)
    2. check for number of characters
    """

    pages_scanned = []
    total_stats = []    

    for i, page in enumerate(doc):
        
        if i >= try_first_n_pages:
            break        

        if display_page_bbox:
            display_page_with_bbox(page)

        # determine if scanned by size of largest image in the page and the page size
        scanned1, image_stats = is_scanned_pdf_by_size(page, size_threshold=0.9)

        # if suspected scanned, see if the largest image is blank (if blank, it's likely background)
        if scanned1:
            images = page.get_images()                
            largest_image = max(images, key=lambda x: x[2] * x[3])            
            xref = largest_image[0]            
            base_image = doc.extract_image(xref)
            # image_bytes = base_image["image"]            
            # image_ext = base_image["ext"]            
            image = Image.open(BytesIO(base_image["image"])) # load w/ PIL
            if display_image:
                image.show()            
            is_blank, pix_std = is_blank_image(image, std_threshold=std_pix_threhold)  # check whether black by std of pixels
            if is_blank:
                scanned1 = False

        scanned2, num_chars_n_imgs = is_scanned_pdf_by_counts(
            page, max_text_length=max_text_length, min_num_images=1
            )
        
        num_chars_n_imgs.update(image_stats)
        num_chars_n_imgs.update({'pix_std': pix_std})
        page_stats = num_chars_n_imgs

        is_page_scanned = True if all([scanned1, scanned2]) else False

        pages_scanned.append(is_page_scanned)
        total_stats.append(page_stats)

    is_scanned = True if any(pages_scanned) else False  
    
    if return_stats:
        return is_scanned, pd.DataFrame(total_stats)
    else:
        return is_scanned


def is_scanned_pdf_by_counts(
        page: fitz.Page,
        max_text_length: int=100,
        min_num_images: int=1
        ):
    
    num_chars = get_text_length(page)
    num_images = get_num_images(page)

    if num_chars < max_text_length and num_images >= min_num_images:
        return True, {'num_chars': num_chars, 'num_images': num_images}
    else:
        return False, {'num_chars': num_chars, 'num_images': num_images}


def is_scanned_pdf_by_size(
        page: fitz.Page,
        size_threshold: float=0.9
        ):

    # Get the list of images on the current page
    image_info_list = page.get_image_info()

    if len(image_info_list) > 0:

        # Calculate the page area
        page_area = page.rect.width * page.rect.height

        largest_image_area = 0
        largest_width = 0
        largest_height = 0
        for img_info in image_info_list:

            # Get the image width and height (x1-x0, y1-y0)        
            img_width = img_info["bbox"][2]-img_info["bbox"][0] 
            img_height = img_info["bbox"][3]-img_info["bbox"][1]

            # Calculate the image area
            img_area = img_width * img_height

            if img_area > largest_image_area:
                largest_image_area = img_area
                largest_width = img_width
                largest_height = img_height

        # Check if the largest image covers at least the size_threshold of the page
        if largest_image_area / page_area >= size_threshold:
            return True, {"largest_image_dimension": (largest_width, largest_height), "page_dimension": (page.rect.width, page.rect.height)}
        
        # If none of the pages have a large enough image, it's likely a structured PDF
        return False, {"largest_image_dimension": (largest_width. largest_height), "page_dimension": (page.rect.width, page.rect.height)}
    
    else:
        print("No images found on page")
        return None, {}


def is_blank_image(image: Image.Image, std_threshold: float=5):
    
    # Convert the image to grayscale
    gray_image = image.convert("L")

    # Convert the image to a numpy array and calculate the standard deviation
    img_array = np.asarray(gray_image)
    std_pix = np.std(img_array)

    # If the standard deviation is close to zero, the image is likely blank
    return std_pix < std_threshold, std_pix

# display page with transparent rectangle overlay
def display_page_with_bbox(page: fitz.Page, save: bool=False, save_path: str=None):
    
    text_blocks = page.get_text("blocks")  
    if len(text_blocks) > 0:  
        for block in text_blocks:
            page.add_rect_annot(block[:4])
    else:
        print('No text found')

    image_blocks = page.get_image_info()
    if len(image_blocks) > 0:
        for block in image_blocks:
            page.add_rect_annot(block["bbox"])
    else:
        print('No image found')

    # display testpage
    page_image = Image.open(BytesIO(page.get_pixmap().tobytes()))
    page_image.show()

    # save image
    if save:
        page_image.save(save_path)


def parse_pdf(doc_path, page_num):

    # read pdf
    # doc_path = '/home/lstm/Github/pdf2html/DATA_sample/보험/sample_insurance_policy.pdf'
    doc = fitz.open(doc_path)
    page = doc[page_num]

    # get text
    parsed = page.get_text()

    # other attributes
    
    return parsed
    
    
def pdf2image(doc_path, page_num, out_path):
    
    # open pdf doc and convert to image. image path is in parent folder
    # docpath = '/home/lstm/Github/pdf2html/부산지법_2018나55364_판결서.pdf'
    # docpath = '/home/lstm/Github/pdf2html/DATA_sample/보험/sample_insurance_policy.pdf'
    doc = fitz.open(doc_path)
    # page 2 from doc
    page = doc[page_num]
    pix = page.get_pixmap()
    pix.save(out_path)

    return pix


def get_block_bbox(page):
    """Extracts block bounding boxes from a PDF page.
    :param page: PDF page to iterate through
    :type page: <class 'fitz.fitz.Page'>
    :rtype: list
    :return: list of block bounding boxes
    """
    blocks = page.get_text("dict")["blocks"]
    bbox = []
    for b in blocks:  # iterate through the text blocks
        if b['type'] == 0:  # block contains text (1 is for imagesblock)
            bbox.append(b['bbox'])

    # create image with bounding boxes    
    for b in bbox:
        page.draw_rect(b, color=fitz.utils.getColor('red'), overlay=True)

    pixmap = page.get_pixmap()
    pixmap.save('bbox.png')    

    return bbox, pixmap
    

def get_fonts_meta(doc, granularity=False):
    """Extracts fonts and their usage in PDF documents.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param granularity: also use 'font', 'flags' and 'color' to discriminate text
    :type granularity: bool
    :rtype: [(font_size, count), (font_size, count}], dict
    :return: most used fonts sorted by count, font style information
    """
    styles = {}
    font_counts = {}
    bbox = {}

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text (1 is for imagesblock)
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if granularity:
                            identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                            # s['flags'] (data is often inaccurate)
                            # bit 0: superscripted – not a font property, detected by MuPDF code.
                            # bit 1: italic
                            # it 2: serifed
                            # bit 3: monospaced
                            # bit 4: bold
                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                  'color': s['color']}
                            if identifier not in bbox:
                                bbox[identifier] = []
                                bbox[identifier] = bbox[identifier] + [s['bbox']]
                                s['bbox']
                            else:
                                bbox[identifier] = bbox[identifier] + [s['bbox']]
                        else:
                            identifier = "{0}".format(s['size'])
                            styles[identifier] = {'size': s['size'], 'font': s['font']}

                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage

    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)

    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")

    return font_counts, styles, bbox


def font_tags(font_counts, styles):
    """Returns dictionary with id keys and tags as value.
    :param font_counts: (font_size, count) for all fonts occuring in document
    :type font_counts: list
    :param styles: all styles found in the document
    :type styles: dict
    :rtype: dict
    :return: all element tags based on font-sizes
    """





    
    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
    p_size = p_style['size']  # get the paragraph's size

    # sorting the font sizes high to low, so that we can append the right integer to each tag 
    font_sizes = []
    for (font_size, count) in font_counts:
        font_sizes.append(float(font_size))
    font_sizes.sort(reverse=True)

    # aggregating the tags for each font size
    idx = 0
    size_tag = {}
    for size in font_sizes:
        idx += 1
        if size == p_size:
            idx = 0
            size_tag[size] = '<p>'
        if size > p_size:
            size_tag[size] = '<h{0}>'.format(idx)
        elif size < p_size:
            size_tag[size] = '<s{0}>'.format(idx)

    return size_tag