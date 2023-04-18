# %%
import fitz, cv2
import os, sys, json
from PIL import Image
from io import BytesIO
import numpy as np
from paddleocr import PPStructure,draw_structure_result,save_structure_res

# add path
sys.path.append(os.path.abspath('./helpers'))
from helpers.utils_fitz import is_scanned_pdf, display_page_with_bbox # for checking scanned PDF
from helpers.utils_fitz import pdf2image, parse_pdf # for checking layout (table, image, separator, footnote)
    #is_scanned_pdf_by_size, is_scanned_pdf_by_counts, is_blank_image \
    #get_block_bbox, parse_pdf, get_fonts_meta, pdf2image, get_text_length, get_num_images 
from helpers.utils_fitz import get_fonts_meta, get_block_bbox     

# max number of texts to be considered scanned
max_chars = 100

# test-open various formats and review
docpath = './DATA_sample/판례/adobe'
sample_pdf_adobes = os.listdir(docpath)#/대법원_2020도1007_판결서.pdf'
testdocs = sample_pdf_adobes

# docpath = './DATA_sample/판례/scan'
# sample_pdf_scans = os.listdir(docpath)#/서울행정법원 2017구합86125.pdf'
# testdocs = sample_pdf_scans

# docpath = './DATA_sample/판례/png'
# sample_pdf_images = os.listdir(docpath)#/image1_0.png'
# testdocs = sample_pdf_images

# %% check whether document is scanned image in PDF format

potential_scanned_docs = []
for testdoc in testdocs:
    print(testdoc)
    doc = fitz.open(f'{docpath}/{testdoc}')
    scanned, df_stats = is_scanned_pdf(
        doc, 
        max_text_length=max_chars,
        std_pix_threhold=5, 
        display_page_bbox=True,
        display_image=True,
        # display_stats=True,
        return_stats=True,
        try_first_n_pages=3
        )
    potential_scanned_docs.append(testdoc) if scanned else None
    print(df_stats)

potential_scanned_docs = list(set(potential_scanned_docs))

# %% (for PDF format) Get doc structure (i.e. headers & texts) & check for table, image, separator, and footnote
# ============ need custom  trained model eventurally
# ============ using pretrained Paddle model for now (w/ much longer latency)

# define parameters
# docpath = './DATA_sample/판례/adobe'
# docpath = './DATA_sample/판례/scan'
docpath = './DATA_sample/보험/pdf'
sample_pdf_adobes = os.listdir(docpath)#/대법원_2020도1007_판결서.pdf'
testdocs = sample_pdf_adobes

save_folder = './output_layout'

# ================ get layout (w/ Paddle - need to convert to PIL first)
# other options: Lilt, Detectron2, LayoutParser
table_engine = PPStructure(show_log=True) #, image_orientation=True)
font_path = '/home/lstm/Github/pdf2html/PP/NotoSansKR-Regular.otf' # needed for PaddleOCR main script

docs_dict = {}
for testdoc in testdocs:
    print(testdoc)
    doc = fitz.open(f'{docpath}/{testdoc}')

    page_dict = {}
    for page_id, page in enumerate(doc):

        # get pixmap
        page_image = page.get_pixmap()

        # convert to np array
        pic = np.array(Image.open(BytesIO(page_image.tobytes())))        

        # get layout
        result = table_engine(pic)

        # show result
        if page_id == 0 or page_id % 5 == 0:            
            im_show = draw_structure_result(pic, result,font_path=font_path)
            im_show = Image.fromarray(im_show)
            im_show.show()

        # reorder result by bbox[1] (y1), then bbox[0] (x1)
        result = sorted(result, key=lambda x: (x['bbox'][1], x['bbox'][0]))        

        block_dict = {}
        for block_id, block in enumerate(result):
            block_dict[block_id] = {'bbox': block['bbox'], 'type': block['type']}

        page_dict[page_id] = block_dict

    docs_dict[testdoc] = page_dict  # key structure: docname -> page -> block: bbox, type
    # save docts_dict as json file, using json library, with indent=4
    with open(f'{save_folder}/{testdoc}.json', 'w') as f:
        json.dump(docs_dict, f, indent=4)
        
    
    
        
# %%
# take pages 10 ~ 13 of pdf and save as pdf
doc = fitz.open('/mnt/c/users/jp/downloads/myanycar_personal.pdf')
doc.select(range(55, 56))
doc.save('/mnt/c/users/jp/downloads/sample_portrait.pdf')


# %%
        







# get fonts metadata
page = doc[0]
block_bbox, page_image = get_block_bbox(page)
Image.open(BytesIO(page_image.tobytes()))
# %%
fonts_meta, style, bbox = get_fonts_meta(doc, granularity=True)
# %%
df_count = pd.DataFrame(fonts_meta, columns=['id', 'count'])

df_style = pd.DataFrame(style).T
df_style.index.name = 'id'
df_style = df_style.reset_index()

df_bbox_list = pd.DataFrame({key: pd.Series([value]) for key, value in bbox.items()}).T
df_bbox_cell = pd.DataFrame.from_dict(bbox, orient='index').T

df = pd.merge(df_count, df_style, on='id', how='left')






# get OCR







# determine headers
# by layout (use PPStructure or equivalent)
# by font style / size (use PDF font metadata)
# by context / content (use LLM)




# extract structure






for testdoc in testdocs:
    print(testdoc)
    doc = fitz.open(f'{docpath}/{testdoc}')
    # get layout
    result = parse_pdf(doc, table_engine, save_folder, testdoc)
    # check layout
    display_page_with_bbox(doc, result, display_image=True, display_table=True, display_separator=True, display_footnote=True)
# img_path = '/home/lstm/Github/pdf2html/누락판례/png/image_footnote.png'
img_path = '/home/lstm/Github/pdf2html/DATA_sample/보험/image_insurance_p45.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

font_path = '/home/lstm/Github/pdf2html/PP/NotoSansKR-Regular.otf' # PaddleOCR
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')

# read pdf
doc_path = '/home/lstm/Github/pdf2html/DATA_sample/보험/sample_insurance_policy.pdf'
doc = fitz.open(doc_path)
page43 = doc[42]


# %% (for scanned PDF) check for table, image, separator, and footnote












# check of table



# check for image



# check for separtor
# ============ need to train model ===============


# chck for footnote



# Open the PDF document
file_path = "path/to/your/pdf_file.pdf"
doc = fitz.open(file_path)

# Display the first page with rectangle overlays around all detected text
page_number = 0
page = doc[page_number]
display_page_with_text_overlay(testpage)
