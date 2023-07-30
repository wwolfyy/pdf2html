# %% imports
import fitz, cv2
import os, sys, json
from PIL import Image
from io import BytesIO
import numpy as np
from paddleocr import PPStructure,draw_structure_result,save_structure_res
import matplotlib.pyplot as plt
from matplotlib import patches
import html
import base64
# import bs4


# add path
sys.path.append(os.path.abspath('./helpers'))
from helpers.utils_fitz import is_scanned_pdf, display_page_with_bbox # for checking scanned PDF
from helpers.utils_fitz import pdf2image, parse_pdf # for checking layout (table, image, separator, footnote)
    #is_scanned_pdf_by_size, is_scanned_pdf_by_counts, is_blank_image \
    #get_block_bbox, parse_pdf, get_fonts_meta, pdf2image, get_text_length, get_num_images
from helpers.utils_fitz import get_fonts_meta, get_block_bbox

from helpers.utils_other import render_bboxes_2_with_image, parse_layout, parse_texts_page, \
    bbox_area, bbox_intersection, bbox_overlap_percentage, produce_html, check_bits
from helpers.utils_pp import render_PP_layout_from_array


# %% define functions

# function to deal with idiosyncracy (thesis)
def ocr_rules_thesis(
    block,
    tag
):
    pass
    # texts not tagged in layout -- if not on bottom, treat as plain text
    # superscripts
    # references not in footnote location


# %% parameters
# parsers
layout_parser = 'PP'
PP_font_path = '/home/lstm/Github/pdf2html/PP/NotoSansKR-Regular.otf' # PaddleOCR
text_parser = 'pymupdf'  # naver
# doc path
folderpath = "./DATA/training_data/leg_trend/Issue_pdf/"
docindex = 6

# folderpath = "./DATA/cases/" # 2022허4406.pdf"
# docindex = 0

doclist = os.listdir(folderpath)
# print(doclist)

docpath = os.path.join(folderpath, doclist[docindex])
print(doclist[docindex])


# %% read doc
doc = fitz.open(docpath)
# doc = fitz.open(stream=file.read(), filetype='pdf')

rotation = doc.load_page(0).rotation
# if not rotation == 0:
#     st.warning('테스트 서버에서는 세로 형식의 문서만 지원합니다.')
#     st.experimental_rerun()
dimension = (doc[0].get_pixmap().irect[2], doc[0].get_pixmap().irect[3])

# st.session_state['doc'] = doc
# st.session_state['dimension'] = dimension

# %% parse layout & text --> put in parsed_dict

parsed_dict = {}

# with st.spinner('문서 구조를 인식중입니다'):

# loop through pages
for pagenum, page in enumerate(doc):

    #

    # convert to np array
    # pic = np.array(Image.open(BytesIO(page_image.tobytes())))

    # get pixmap and convert to a bytes object
    page_image = page.get_pixmap()
    img_data = page_image.tobytes("png")

    # get layout
    layout_result, layout_image_array = parse_layout(
        parser=layout_parser,
        image_byte=img_data,
        pp_font_path=PP_font_path)

    # get texts (using pymupdf or naver OCR)
    page_dict = parse_texts_page(parser=text_parser, pymupdf_page_object=page, image_byte_for_ocr=None)

    # <<=========================== font attribute detection ===========================
    # superscipt, italic, bold
    # https://pymupdf.readthedocs.io/en/latest/textpage.html#textpagedict

    # get list of blocks from page_dict (corresponds to <p> in html)
    p_list = [page_dict['blocks'][k]['lines'][0] for k in range(len(page_dict['blocks'])) if page_dict['blocks'][k]['type'] == 0]
    # get list of 'spans' and 'bbox' for the span
    span_list_nested = [p_list[k]['spans'] for k in range(len(p_list))]
    # flatten the list
    span_list = [item for sublist in span_list_nested for item in sublist]

    # get flags for superscript spans (where bit 0 of flags is 1)
    #
    # sup_span_list = [
    #     {'text': span_list[k]['text'], 'bbox': span_list[k]['bbox'], 'flags': span_list[k]['flags']}\
    #     for k in range(len(span_list))\
    #     if bin(span_list[k]['flags'])[-1] == '1'
    #     ]
    # find span in page_dict that matches the bbox of the superscript span
    # for sup_span in sup_span_list:

    for k in range(len(page_dict['blocks'])):
        if page_dict['blocks'][k]['type'] == 0:  # for text blocks
            for l in range(len(page_dict['blocks'][k]['lines'])):
                for m in range(len(page_dict['blocks'][k]['lines'][l]['spans'])):

                    print(page_dict['blocks'][k]['lines'][l]['spans'][m]['flags'])
                    page_dict['blocks'][k]['lines'][l]['spans'][m]['is_superscript'] = False
                    page_dict['blocks'][k]['lines'][l]['spans'][m]['is_italic'] = False
                    page_dict['blocks'][k]['lines'][l]['spans'][m]['is_bold'] = False

                    bitcheck = check_bits(page_dict['blocks'][k]['lines'][l]['spans'][m]['flags'])
                    if bitcheck[0]:
                        page_dict['blocks'][k]['lines'][l]['spans'][m]['is_superscript'] = True
                        print('super:', page_dict['blocks'][k]['lines'][l]['spans'][m]['text'])
                    if bitcheck[1]:
                        page_dict['blocks'][k]['lines'][l]['spans'][m]['is_italic'] = True
                        print('italic:', page_dict['blocks'][k]['lines'][l]['spans'][m]['text'])
                    if bitcheck[4]:
                        page_dict['blocks'][k]['lines'][l]['spans'][m]['is_bold'] = True
                        print('bold:', page_dict['blocks'][k]['lines'][l]['spans'][m]['text'])

    # ===================================================================================>>

    # get bboxes by type from layout
    ref_bbox = []  # footnote
    header_bbox = []
    title_bbox = []
    footer_bbox = []
    text_bbox = []
    for i, el in enumerate(layout_result):
        if el['type'] == 'reference':
            ref_bbox = ref_bbox + [el['bbox']]
        if el['type'] == 'header':
            header_bbox = header_bbox + [el['bbox']]
        if el['type'] == 'title':
            title_bbox = title_bbox + [el['bbox']]
        if el['type'] == 'footer':
            footer_bbox = footer_bbox + [el['bbox']]
        if el['type'] == 'text':
            text_bbox = text_bbox + [el['bbox']]

    # deal with overlapping layout bboxes
    # [TBD]

    # <<===================== for refernece miss-tagging in layout parsing =================
    # check if reference bbox is above anything other than footer (change to text if so)
    del_idx = []
    if len(ref_bbox) > 0:
        for r, ref_b in enumerate(ref_bbox):
            for i, el in enumerate(layout_result):
                if el['type'] not in ['footer', 'reference']:
                    print(el['bbox'][1], ref_bbox[r][1])
                    # if both start and end of y-axis of bbox are below those of reference bbox
                    if all([el['bbox'][1] > ref_bbox[r][1], el['bbox'][3] > ref_bbox[r][3]]):
                        print('reference bbox is above something other than footer or reference')
                        # layout_result[i]['type'] = 'text'
                        del_idx = del_idx + [r]
    del_idx = list(set(del_idx))
    ref_bbox = [ref_bbox[i] for i in range(len(ref_bbox)) if i not in del_idx]

    # change "reference" to "text" for bbox that are not footnote
    for i, el in enumerate(layout_result):
        if el['type'] == 'reference':
            if el['bbox'] not in ref_bbox:
                layout_result[i]['type'] = 'text'
    # ======================================================================================>>

    # <<=================== for tagging text parsing with layout parsing =====================
    # mark if a block from text parser is a footnote (i.e. ref), etc. (based on overlapping area)
    # i.e. if sum of overlapping areas between a block and reference bbox is > 100
    for i in range(len(page_dict['blocks'])):
        page_dict['blocks'][i]['is_footnote'] = False
        page_dict['blocks'][i]['is_header'] = False
        page_dict['blocks'][i]['is_title'] = False
        page_dict['blocks'][i]['is_footer'] = False

        block_bbox = page_dict['blocks'][i]['bbox']

        # get overlap percentage with reference bbox
        for r, ref_b in enumerate(ref_bbox):
            aa, bb = bbox_overlap_percentage(block_bbox, ref_bbox[r])
            if aa + bb > 100:
                page_dict['blocks'][i]['is_footnote'] = True
                continue
        # header bbox
        for r, header_b in enumerate(header_bbox):
            aa, bb = bbox_overlap_percentage(block_bbox, header_bbox[r])
            if aa + bb > 100:
                page_dict['blocks'][i]['is_header'] = True
                continue

        for r, title_b in enumerate(title_bbox):
            aa, bb = bbox_overlap_percentage(block_bbox, title_bbox[r])
            if aa + bb > 100:
                page_dict['blocks'][i]['is_title'] = True
                continue

        for r, footer_b in enumerate(footer_bbox):
            aa, bb = bbox_overlap_percentage(block_bbox, footer_bbox[r])
            if aa + bb > 100:
                page_dict['blocks'][i]['is_footer'] = True
                continue

    # <<=========================== for other tagging ========================================
    # header
    # for i in range(len(page_dict['blocks'])):
    #     page_dict['blocks'][i]['is_header'] = False
    #     block_bbox = page_dict['blocks'][i]['bbox']

    #     # get overlap percentage with reference bbox
    #     overlap = []
    #     for r, ref_b in enumerate(ref_bbox):
    #         aa, bb = bbox_overlap_percentage(block_bbox, ref_bbox[r])
    #         if aa + bb > 0:
    #             page_dict['blocks'][i]['is_footnote'] = True
    #             continue
    # text, header, footer, title
    # PASS
    # ======================================================================================>>\

    tmp_dict = {
        'layout': layout_result,  # parsed results for layout
        'texts': page_dict,  # parsed results for texts
        'imagebytes_page': img_data, # b64 string
        'imagearray_layout': layout_image_array  # np array
        }

    parsed_dict[pagenum] = tmp_dict

# %% display bboxes and layout labels
for i in parsed_dict.keys():

    bboxes1 = [parsed_dict[i]['layout'][j]['bbox'] for j in range(len(parsed_dict[i]['layout']))]
    bboxes2 = [parsed_dict[i]['texts']['blocks'][j]['bbox'] for j in range(len(parsed_dict[i]['texts']['blocks']))]
    print(bboxes2)
    bbox_groups = [bboxes1, bboxes2]
    labels = [parsed_dict[i]['layout'][j]['type'] for j in range(len(parsed_dict[i]['layout']))]
    label_index = 0

    image_byte = parsed_dict[i]['imagebytes_page']
    layout_image_array = parsed_dict[i]['imagearray_layout']

    render_bboxes_2_with_image(
        image_byte=image_byte,

        bbox_groups=bbox_groups,
        labels=labels,
        label_index=label_index,
        color=['red', 'blue'],
        fontsize=4,
        # display_labels=False,
        alpha=0.5, thickness=1, dpi=200)

    render_PP_layout_from_array(layout_image_array)

# %% produce html
html_div_main = ''
html_div_footnote = ''
html_div_header = ''
html_div_title = ''
html_div_footer = ''
for key, val in parsed_dict.items():
    # sort val by bbox[1] (y1), then by bbox[0] (x1)
    val['texts']['blocks'] = sorted(val['texts']['blocks'], key=lambda k: (k['bbox'][1], k['bbox'][0]))
    for j, block in enumerate(val['texts']['blocks']):
        tag_para_left = str(round(block['bbox'][0], 2))
        if block['type'] == 0: # text block -- treat each line as a paragraph
            if not block['is_footnote']:
                for k, para in enumerate(block['lines']):
                    html_span = ''
                    for l, span in enumerate(block['lines'][k]['spans']):
                        # print(span['is_superscript'], span['text'])
                        # if span['is_superscript']:
                            # print('superscript', span['text'])


                        # mandatory tags
                        # tag_span_fontsize = str(round(span['size'], 2))
                        # tag_span_left = str(round(span['bbox'][0], 2))

                        # beginning and endding tags for span
                        # tag_span_begin = f'<span id="span_{i}_{j}_{k}_{l}" style="font-size: {tag_span_fontsize}px; margin-left: {tag_span_left}px;">'
                        tag_span_begin = f'<span id="span_{i}_{j}_{k}_{l}">'
                        tag_span_end = '</span>'

                        # optional tags
                        if span['is_superscript']:
                            # print('superscript', span['text'])
                            tag_span_begin = tag_span_begin + '<sup>'
                            tag_span_end = '</sup>' + tag_span_end
                        if span['is_italic']:
                            tag_span_begin = tag_span_begin + '<i>'
                            tag_span_end = '</i>' + tag_span_end
                        if span['is_bold']:
                            tag_span_begin = tag_span_begin + '<b>'
                            tag_span_end = '</b>' + tag_span_end

                        # string for span
                        html_span = html_span + tag_span_begin + span['text'] + tag_span_end

                    # headers and footers (headers: take out if not first page; footers: take out)
                    if block['is_header']:
                        if key == 0:
                            html_div_main = html_div_main + '<p style=display: inline-block;margin-left: {tag_para_left}px;>' + html_span + '</p>' + '\n'
                        else:
                            html_div_header = html_div_header + '<p>' + html_span + '</p>' + '\n'
                    elif block['is_footer']:
                        html_div_footer = html_div_footer + '<p>' + html_span + '</p>' + '\n'
                    else:
                        # main div
                        if block['is_title']:
                            html_div_main = html_div_main + '<h3 style=margin-left: {tag_para_left}px;>' + html_span + '</h1>' + '\n'
                        else:
                            html_div_main = html_div_main + '<p style=display: inline-block;margin-left: {tag_para_left}px;>' + html_span + '</p>' + '\n'

            elif block['is_footnote']:
                for k, para in enumerate(block['lines']):
                    html_span = ''
                    for l, span in enumerate(block['lines'][k]['spans']):
                        # mandatory tags
                        tag_span_fontsize = str(round(span['size'], 2))
                        tag_span_left = str(round(span['bbox'][0], 2))

                        # beginning and endding tags for span
                        # tag_span_begin = f'<span id="span_{i}_{j}_{k}_{l}" style="font-size: {tag_span_fontsize}px; margin-left: {tag_span_left}px;">'
                        tag_span_begin = f'<span id="span_{i}_{j}_{k}_{l}">'
                        tag_span_end = '</span>'

                        # optional tags
                        if span['is_superscript']:
                            tag_span_begin = tag_span_begin + '<sup>'
                            tag_span_end = '</sup>' + tag_span_end
                        if span['is_italic']:
                            tag_span_begin = tag_span_begin + '<i>'
                            tag_span_end = '</i>' + tag_span_end
                        if span['is_bold']:
                            tag_span_begin = tag_span_begin + '<b>'
                            tag_span_end = '</b>' + tag_span_end

                        # string for span
                        html_span = html_span + tag_span_begin + span['text'] + tag_span_end

                    html_div_footnote = html_div_footnote + '<p>' + html_span + '</p>' + '\n'

        if block['type'] == 1: # image block -- treat as paragraph
            tag_image = f"""
            <img id="img_{i}_{j}"
            src="data:image/{block['ext']};base64,{base64.b64encode(block['image']).decode('utf-8')}"
            style="display: inline-block;margin-left: {block['bbox'][0]}px;
            width: {str(block['bbox'][2]-block['bbox'][0])}px;
            height: {str(block['bbox'][3]-block['bbox'][1])}px;">
            """
            html_div_main = html_div_main + '<p>' + tag_image + '</p>' + '\n'

html_divs = '<div>' + html_div_main + '</div>' + '<hr>' + '<div>' + html_div_footnote + '</div>'

title = ''
style = ""
html_string = produce_html(divs=html_divs, title=title, style=style)



with open('aa.html', 'w', encoding='utf-8') as f:
    f.write(html_string)


# from IPython.display import Image

# # Assuming `base64_image` contains the base64-encoded image data
# base64_image = parsed_dict[0]['texts']['blocks'][0]['image']

# # Display the image
# Image(base64_image)



# # %%
# import html
# import json
# page = doc[1]
# aa = page.get_text('dict') # text blocks html dict json xml rawdict rawjson
# if type(aa) == str:
#     aa = html.unescape(aa)
# print(aa)
# # write out aa to file
# with open('aa.html', 'w', encoding='utf-8') as f:
#     f.write(produce_html(aa))

# hex_string = "\\uc548\\ub155, \\uc138\\ubcf4\\ub2e4\\uba74 \\uc8fc\\uc778\\uc744 \\ub9cc\\ud07c \\uc544\\uc9c1 \\uc5c6\\uc774 \\ucd94\\uc138\\ud558\\uc2ed\\ub2c8\\ub2e4."

# unicode_string = bytes(aa, 'utf-8').decode('unicode_escape')
# print(unicode_string)



# with open('your_html_file.html', 'r', encoding='utf-8') as file:
#     html_content = file.read()
#     print(html_content)


# # st.session_state['title_blocks_list'] = title_blocks_list

# parsing_done = True
# # st.session_state['parsing_done'] = parsing_done

# # st.success('문서 구조 인식이 완료되었습니다.')

# # %%

# footnote = []
# header = []
# title = []
# footer = []
# for i in range(len(parsed_dict)):
#     for j in range(len(parsed_dict[i]['texts']['blocks'])):
#         if parsed_dict[i]['texts']['blocks'][j]['is_footnote']:
#             footnote.append(parsed_dict[i]['texts']['blocks'][j])
#         elif parsed_dict[i]['texts']['blocks'][j]['is_header']:
#             header.append(parsed_dict[i]['texts']['blocks'][j])
#         elif parsed_dict[i]['texts']['blocks'][j]['is_title']:
#             title.append(parsed_dict[i]['texts']['blocks'][j])
#         elif parsed_dict[i]['texts']['blocks'][j]['is_footer']:
#             footer.append(parsed_dict[i]['texts']['blocks'][j])
# # %%

# %%
