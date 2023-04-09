# %% ---------------------------------------------------------------------------
# prep 

import layoutparser as lp
import cv2
import pandas as pd

# %% ---------------------------------------------------------------------------
# define parameters

imagepath = '/home/lstm/Github/pdf2html/누락판례/png/image61_0.png'
image = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2RGB)  # read image and convert to RGB
    # image[..., ::-1] also works

print(image.shape)
h, w, d = image.shape

# %% ---------------------------------------------------------------------------
# set up LP model

# model_path = 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config'  # NG
# model_path = 'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config'  # somewhat better
# label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}

model_path = 'lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config'  # somewhat works
label_map={1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"}

# model_path = 'lp://HJDataset/mask_rcnn_R_50_FPN_3x/config'
# model_path = 'lp://HJDataset/retinanet_R_50_FPN_3x/config'
# label_map={1:"Page Frame", 2:"Row", 3:"Title Region", 4:"Text Region", 5:"Title", 6:"Subtitle", 7:"Other"}

# model_path = 'lp://TableBank/faster_rcnn_R_50_FPN_3x/config'  # works
# model_path = 'lp://TableBank/faster_rcnn_R_101_FPN_3x/config'  # works
# label_map={0: "Table"}

model = lp.Detectron2LayoutModel(model_path,
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                 label_map=label_map)

# model = lp.AutoLayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
#                                  extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
#                                  label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

# model = ld.PaddleOCRLayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',

# Model Zoo: https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html

# %% ---------------------------------------------------------------------------
# detect layout

layout = model.detect(image) 
lp.draw_box(image, layout, box_width=3)  # Show the detected layout of the input image

# %% ---------------------------------------------------------------------------
# remove blocks enclosed by another block

block_to_remove = []
for block in layout:
    for another_block in layout:
        if block is another_block:
            continue
        else:
            if block.is_in(another_block):                
                block_to_remove.append(block)

for block in block_to_remove:
    layout.remove(block)

lp.draw_box(image, layout, box_width=3)

# %% ---------------------------------------------------------------------------
# get blocks and group by type

block_types = set([b.type for b in layout])
print(block_types)

text_blocks = lp.Layout([b for b in layout if 'text' in b.type.lower()])
figure_blocks = lp.Layout([b for b in layout if 'figure' in b.type.lower()])
table_blocks = lp.Layout([b for b in layout if 'table' in b.type.lower()])
title_blocks = lp.Layout([b for b in layout if 'title' in b.type.lower()])
# separator_blocks = lp.Layout([b for b in layout if b.type=='Separator'])

# %% ---------------------------------------------------------------------------
# set up OCR

# ocr_agent = lp.TesseractAgent(languages='kor')

# lp.is_paddle_available()
# ocr_agent = lp.paddleocr_agent(languages='eng')

ocr_agent = lp.GCVAgent.with_credential("/home/lstm/.gcp/koreacaselaw-ffe91e33f191.json", languages = ['en', 'ko'])

# %% --------------------------------------------------------------------------
# read in texts in text & title blocks
# then put in block.text object (of text_blocks) 

for block in title_blocks:
    segment_image = (block
                       .pad(left=5, right=5, top=5, bottom=5)
                       .crop_image(image))
        # add padding in each image segment can help improve robustness
    
    text = ocr_agent.detect(segment_image)
    block.set(text=text, inplace=True)
	
for block in text_blocks:
    segment_image = (block
                       .pad(left=5, right=5, top=5, bottom=5)
                       .crop_image(image))        

    text = ocr_agent.detect(segment_image)
    block.set(text=text, inplace=True)

for txt in text_blocks.get_texts():
    print(txt, end='\n---\n')

for txt in title_blocks.get_texts():
    print(txt, end='\n---\n')    

# %% ---------------------------------------------------------------------------
# crop image blocks from figure_blocks, table_blocks, separator_blocks, and 
# 1) assign figure index 2) save as file w/ index as figure name

import matplotlib.pyplot as plt
%matplotlib inline

for idx, block in enumerate(figure_blocks):   

    block.set(id=idx, inplace=True)     
        
    plt.imshow(block.crop_image(image))
    plt.axis('off')
    plt.savefig('image_' + str(idx) + '.png', bbox_inches='tight')    
    
# plt.imshow(figure_blocks.crop_image(image)[0])  # this crops all image blocks from image

for idx, block in enumerate(table_blocks):   

    block.set(id=idx, inplace=True)     
        
    plt.imshow(block.crop_image(image))
    plt.axis('off')
    plt.savefig('table_' + str(idx) + '.png', bbox_inches='tight')   

for idx, block in enumerate(separator_blocks):   

    block.set(id=idx, inplace=True)     
        
    plt.imshow(block.crop_image(image))
    plt.axis('off')
    plt.savefig('sep_' + str(idx) + '.png', bbox_inches='tight')   
		
# %% ---------------------------------------------------------------------------
# put everything in dictionary, and order by upper left coordinate

all_blocks = title_blocks + text_blocks + figure_blocks + table_blocks# + separator_blocks
all_blocks = all_blocks.sort(key = lambda b:b.coordinates[1]).to_dataframe()
all_blocks.rename(columns={'text':'text_block'}, inplace=True)

# convert each text region (i.e. collection of text blocks) to paragraph of lines and words
def region_to_paragraph(region):

========================= from here 
    
    if type(region)==lp.elements.layout.Layout:
        return region.get_text().replace('






all_blocks['text'].apply(
    lambda x: x.to_dict() if type(x)==lp.elements.layout.Layout else {}
    ).apply(pd.Series)[
    'blocks'
    ].apply(pd.Series)

pd.DataFrame(all_blocks['text_block'].apply)
all_blocks['text'] = all_blocks['text_block'].apply(
    lambda x: pd.Series(x.to_dict()) if type(x)==lp.elements.layout.Layout else x)

print('block keys:', all_blocks.keys())
all_blocks['blocks']
print('text block keys:', all_blocks['text'].keys())

# %% ---------------------------------------------------------------------------
# convert to better-structured key-value pair

all_blocks = all_blocks.to_dataframe()


dict_block = {}
for idx, item in enumerate(all_blocks):
    print(item.to_dict()['x_1'])
        
    entry = {}
    entry['coordiates'] = item.coordinates    
    entry['type'] = item.type

    if item.type == 'Text':
        colnames = ['x0', 'yo', 'x1', 'y1', 'x3', 'y3', 'x4', 'y4']
        points_2frame = pd.DataFrame(item.text.to_dataframe().points.to_list(), columns=colnames)
        rest_2frame = item.text.to_dataframe().set_index('id').drop(columns=['points'])       

        df_text = pd.concat([points_2frame, rest_2frame], axis=1) 


 
    if item.type == 'Figure':
        entry['filename'] = 'image_' + str(item.id) + '.png'
        entry['figure_id'] = item.id
    dict_block[idx] = entry
	
# dict_block

# %% ---------------------------------------------------------------------------
# reconstruct HTML

import dominate
from dominate.tags import *

doc = dominate.document(title='PDf to HTML tester')

with doc.head:    
    meta(charset="utf-8")
    link(rel='stylesheet', href='style.css')   

with doc:
    with div(id='header').add(ol()):
        for i in ['home', 'about', 'contact']:
            li(a(i.title(), href='/%s.html' % i))

    with div():
        attr(cls='body')
        p('Lorem ipsum..')

print(doc)


# convert series of list to matrix
# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python


