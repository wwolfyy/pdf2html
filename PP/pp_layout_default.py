import os, sys
import cv2
import fitz
from paddleocr import PPStructure,draw_structure_result,save_structure_res
from PIL import Image

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

from helpers.utils_fitz import pdf2image, parse_pdf

# get layout
table_engine = PPStructure(table=False, ocr=False, show_log=True) #, image_orientation=True)

save_folder = './output'
# img_path = '/home/lstm/Github/pdf2html/누락판례/png/image_footnote.png'
img_path = '/home/lstm/Github/pdf2html/DATA_sample/보험/png/image_insurance_p45.png'
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
