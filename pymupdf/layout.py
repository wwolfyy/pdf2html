# %%
import os, sys
from io import BytesIO
from operator import itemgetter
from PIL import Image

import fitz
import pandas as pd


p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

from helpers.utils_fitz import get_fonts_meta, get_block_bbox 

# read pdf
doc_path = '/home/lstm/Github/pdf2html/DATA_sample/판례/adobe/대법원_2020도1007_판결서.pdf'
doc = fitz.open(doc_path)

# display bbox on page 1
page = doc[0]
block_bbox, page_image = get_block_bbox(page)
Image.open(BytesIO(page_image.tobytes()))

# save image
# page_image.save('page_image.png')

# %%
# get font metadata
fonts_meta, style, bbox = get_fonts_meta(doc, granularity=True)

df_count = pd.DataFrame(fonts_meta, columns=['id', 'count'])

df_style = pd.DataFrame(style).T
df_style.index.name = 'id'
df_style = df_style.reset_index()

df_bbox_list = pd.DataFrame({key: pd.Series([value]) for key, value in bbox.items()}).T
df_bbox_cell = pd.DataFrame.from_dict(bbox, orient='index').T

df = pd.merge(df_count, df_style, on='id', how='left')

# %%

