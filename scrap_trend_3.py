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
import re
from typing import Dict

# add path
sys.path.append(os.path.abspath('..'))
from helpers.utils_fitz import is_scanned_pdf, display_page_with_bbox # for checking scanned PDF
from helpers.utils_fitz import pdf2image, parse_pdf # for checking layout (table, image, separator, footnote)
    #is_scanned_pdf_by_size, is_scanned_pdf_by_counts, is_blank_image \
    #get_block_bbox, parse_pdf, get_fonts_meta, pdf2image, get_text_length, get_num_images
from helpers.utils_fitz import get_fonts_meta, get_block_bbox

from helpers.utils_other import render_bboxes_2_with_image, parse_layout, parse_texts_page, \
    bbox_area, bbox_intersection, bbox_overlap_percentage, produce_html, check_bits
from helpers.utils_pp import render_PP_layout_from_array


# %% define functions

def convert_pdf(pdf_bytes: bytes) -> Dict:
    # parameters
    # parsers
    layout_parser = 'PP'
    PP_font_path = '../NotoSansKR-Regular.otf' # PaddleOCR
    text_parser = 'pymupdf'  # naver
    # doc path
    # folderpath = "./DATA/training_data/leg_trend/Issue_pdf/"
    # docindex = 6

    # folderpath = "./DATA/cases/" # 2022허4406.pdf"
    # docindex = 0

    # doclist = os.listdir(folderpath)
    # print(doclist)

    # docpath = os.path.join(folderpath, doclist[docindex])
    # print(doclist[docindex])

    # read doc
    # doc = fitz.open(docpath)
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')

    rotation = doc.load_page(0).rotation
    # if not rotation == 0:
    #     st.warning('테스트 서버에서는 세로 형식의 문서만 지원합니다.')
    #     st.experimental_rerun()
    dimension = (doc[0].get_pixmap().irect[2], doc[0].get_pixmap().irect[3])

    # st.session_state['doc'] = doc
    # st.session_state['dimension'] = dimension

    # parse layout & text --> put in parsed_dict

    parsed_dict = {}

    # with st.spinner('문서 구조를 인식중입니다'):

    # loop through pages
    for pagenum, page in enumerate(doc):
        # page = doc[1]    #

        # convert to np array
        # pic = np.array(Image.open(BytesIO(page_image.tobytes())))

        # get pixmap and convert to a bytes object
        page_image = page.get_pixmap()
        img_data = page_image.tobytes("png")

        # get layout
        layout_result, layout_image_array = parse_layout(
            parser=layout_parser,
            image_byte=img_data,
            pp_font_path=PP_font_path
            )

        # visually check layout
        # render_PP_layout_from_array(layout_image_array)

        # get texts
        page_dict = parse_texts_page(
            parser=text_parser,
            pymupdf_page_object=page,
            image_byte_for_ocr=None,
            parse_option='dict'
            )

        # get all image blocks and text lines from pge_dict and sort in order of
        # (y coordinate, x coordinate) of top left corner
        line_list = []
        for idx, el in enumerate(page_dict['blocks']):
            if el['type'] == 1:  # image blocks
                tmp_line = {
                    'type': 'image',
                    'tag': '<img>',
                    'bbox': el['bbox'],
                    'line': el['image']
                    }
                line_list = line_list + [tmp_line]
            elif el['type'] == 0:  # text blocks
                for line in el['lines']:
                    tagged_text_line = ''
                    org_line_text = ''
                    for span in line['spans']:
                        print(span['flags'])
                        bitcheck = check_bits(span['flags'])
                        if bitcheck[0]:
                            tag = 'sup'  # superscript
                        elif bitcheck[1]:
                            tag = 'i'  # italic
                        elif bitcheck[4]:
                            tag = 'b'  # bold
                        else:
                            tag = ''
                        if tag == '':
                            tagged_text_line += f"{span['text']}"
                        else:
                            tagged_text_line += f"<{tag}>{span['text']}</{tag}>"
                        org_line_text += span['text']
                        tmp_line = {
                            'type': 'text',
                            'tag': '',
                            'bbox': el['bbox'],
                            'line': tagged_text_line
                        }
                    tmp_line['org_line_text'] = org_line_text
                    line_list = line_list + [tmp_line]
            else:
                raise ValueError('Unknown type of layout element')

        # order line list by bbox[1] and then bbox[0]
        line_list = sorted(line_list, key=lambda k: (k['bbox'][1], k['bbox'][0]))

        # <<===================== for refernece miss-tagging in layout parsing =================

        # get reference (footnote) bboxes from layout
        ref_bbox = []
        for i, el in enumerate(layout_result):
            if el['type'] == 'reference':
                ref_bbox = ref_bbox + [el['bbox']]

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

        # get bboxes by type from layout
        ref_bbox = []  # footnote
        header_bbox = []
        title_bbox = []
        footer_bbox = []
        text_bbox = []
        figure_bbox = []
        figure_caption_bbox = []
        table_bbox = []
        for i, el in enumerate(layout_result):
            if el['type'] == 'reference':
                ref_bbox = ref_bbox + [el['bbox']]  # footnote
            elif el['type'] == 'header':
                header_bbox = header_bbox + [el['bbox']]
            elif el['type'] == 'title':
                title_bbox = title_bbox + [el['bbox']]
            elif el['type'] == 'figure':
                figure_bbox = figure_bbox + [el['bbox']]
            elif el['type'] == 'figure_caption':
                figure_caption_bbox = figure_caption_bbox + [el['bbox']]
            elif el['type'] == 'table':
                table_bbox = table_bbox + [el['bbox']]
            elif el['type'] == 'footer':
                footer_bbox = footer_bbox + [el['bbox']]  # footer (page number)
            elif el['type'] == 'text':
                text_bbox = text_bbox + [el['bbox']]
            else:
                raise ValueError('Unknown type of layout element')

        # tag each line for header, reference, footer, and [[ title ]] -- against layout_result
        title_pattern = r'^[A-Za-z0-9Ⅰ-Ⅻ가-힣ㄱ-ㅎ]{1}[^\w\s]*\s*[\uac00-\ud7af]+'
        footer_pattern = r'^[\s\W]{0,3}\d[\s\W]{0,3}$'

        for idx, line in enumerate(line_list):
            print(line)

            # check for header
            if len(header_bbox) > 0:
                check_header = bbox_overlap_percentage(
                    bbox1=line['bbox'], bbox2=header_bbox, comparison_option='bbox_v_list', threshold=0.5
            )
                if len(check_header['indice']) > 0:
                    line_list[idx]['type'] = 'header'
                    continue

            # check for title
            if len(title_bbox) > 0:
                check_title = bbox_overlap_percentage(
                    bbox1=line['bbox'], bbox2=title_bbox, comparison_option='bbox_v_list', threshold=0.5
                )
                if len(check_title['indice']) > 0:
                    if re.match(title_pattern, line['org_line_text']): # verify pattern
                        print('title:', line['org_line_text'])
                        line_list[idx]['type'] = 'title'
                        continue

            # # --- check for title with pattern only ---
            # if bool(re.match(title_pattern, line['org_line_text'])):
            #     print('title:', line['org_line_text'])
            #     line_list[idx]['type'] = 'title'
            #     continue

            # check for reference
            if len(ref_bbox) > 0:
                check_ref = bbox_overlap_percentage(
                    bbox1=line['bbox'], bbox2=ref_bbox, comparison_option='bbox_v_list', threshold=0.5
                )
                if len(check_ref['indice']) > 0:
                    line_list[idx]['type'] = 'reference'
                    continue

            # check for footer
            if len(footer_bbox) > 0:
                check_footer = bbox_overlap_percentage(
                    bbox1=line['bbox'], bbox2=footer_bbox, comparison_option='bbox_v_list', threshold=0.5
                )
                if len(check_footer['indice']) > 0:
                    if all([
                        re.match(footer_pattern, line['org_line_text']),
                        line_list.index(line) == len(line_list) - 1
                        ]):                                     # verify pattern & location
                        print('footer:', line['org_line_text'])
                        line_list[idx]['type'] = 'footer'
                        continue

            # --- check for footer with pattern and location ---
            if line['type'] == 'text':
                if all([
                    re.match(footer_pattern, line['org_line_text']),
                    line_list.index(line) == len(line_list) - 1
                ]):
                    print('footer:', line['org_line_text'])
                    line_list[idx]['type'] = 'footer'
                    continue

            # check for figure -- don't do this -- will cause error
            # if len(figure_bbox) > 0:
            #     check_figure = bbox_overlap_percentage(
            #         bbox1=line['bbox'], bbox2=figure_bbox, comparison_option='bbox_v_list', threshold=0.5
            #     )
            #     if len(check_figure['indice']) > 0:
            #         line_list[idx]['type'] = 'figure'
            #         continue

            # check for figure caption
            if len(figure_caption_bbox) > 0:
                check_figure_cpation = bbox_overlap_percentage(
                    bbox1=line['bbox'], bbox2=figure_caption_bbox, comparison_option='bbox_v_list', threshold=0.5
                )
                if len(check_figure_cpation['indice']) > 0:
                    line_list[idx]['type'] = 'figure_caption'
                    continue

            # check for table
            if len(table_bbox) > 0:
                check_table = bbox_overlap_percentage(
                    bbox1=line['bbox'], bbox2=table_bbox, comparison_option='bbox_v_list', threshold=0.5
                )
                if len(check_table['indice']) > 0:
                    line_list[idx]['type'] = 'table'
                    continue

        # TODO concatenate text lines between titles

        # put in parsed_dict
        parsed_dict[pagenum] = line_list

    # convert image bytestrings to base64 encoded strings
    for key, val in parsed_dict.items():
        for i, line in enumerate(val):
            if line['tag'] == '<img>':
                line['line'] = base64.b64encode(line['line']).decode('utf-8')

    result_json = json.dumps(parsed_dict)

    # return the JSON string as the response content
    return {"result": result_json}


# %% display bboxes and layout labels
# --------------------------------- from here --------------------
# for i in parsed_dict.keys():

#     bboxes1 = [parsed_dict[i]['layout'][j]['bbox'] for j in range(len(parsed_dict[i]['layout']))]
#     bboxes2 = [parsed_dict[i]['texts']['blocks'][j]['bbox'] for j in range(len(parsed_dict[i]['texts']['blocks']))]
#     print(bboxes2)
#     bbox_groups = [bboxes1, bboxes2]
#     labels = [parsed_dict[i]['layout'][j]['type'] for j in range(len(parsed_dict[i]['layout']))]
#     label_index = 0

#     image_byte = parsed_dict[i]['imagebytes_page']
#     layout_image_array = parsed_dict[i]['imagearray_layout']

#     render_bboxes_2_with_image(
#         image_byte=image_byte,

#         bbox_groups=bbox_groups,
#         labels=labels,
#         label_index=label_index,
#         color=['red', 'blue'],
#         fontsize=4,
#         # display_labels=False,
#         alpha=0.5, thickness=1, dpi=200)

#     render_PP_layout_from_array(layout_image_array)

# # %% produce html
# html_div_main = ''
# html_div_footnote = ''
# html_div_header = ''
# html_div_title = ''
# html_div_footer = ''
# for key, val in parsed_dict.items():
#     # sort val by bbox[1] (y1), then by bbox[0] (x1)
#     val['texts']['blocks'] = sorted(val['texts']['blocks'], key=lambda k: (k['bbox'][1], k['bbox'][0]))
#     for j, block in enumerate(val['texts']['blocks']):
#         tag_para_left = str(round(block['bbox'][0], 2))
#         if block['type'] == 0: # text block -- treat each line as a paragraph
#             if not block['is_footnote']:
#                 for k, para in enumerate(block['lines']):
#                     html_span = ''
#                     for l, span in enumerate(block['lines'][k]['spans']):
#                         # print(span['is_superscript'], span['text'])
#                         # if span['is_superscript']:
#                             # print('superscript', span['text'])


#                         # mandatory tags
#                         # tag_span_fontsize = str(round(span['size'], 2))
#                         # tag_span_left = str(round(span['bbox'][0], 2))

#                         # beginning and endding tags for span
#                         # tag_span_begin = f'<span id="span_{i}_{j}_{k}_{l}" style="font-size: {tag_span_fontsize}px; margin-left: {tag_span_left}px;">'
#                         tag_span_begin = f'<span id="span_{i}_{j}_{k}_{l}">'
#                         tag_span_end = '</span>'

#                         # optional tags
#                         if span['is_superscript']:
#                             # print('superscript', span['text'])
#                             tag_span_begin = tag_span_begin + '<sup>'
#                             tag_span_end = '</sup>' + tag_span_end
#                         if span['is_italic']:
#                             tag_span_begin = tag_span_begin + '<i>'
#                             tag_span_end = '</i>' + tag_span_end
#                         if span['is_bold']:
#                             tag_span_begin = tag_span_begin + '<b>'
#                             tag_span_end = '</b>' + tag_span_end

#                         # string for span
#                         html_span = html_span + tag_span_begin + span['text'] + tag_span_end

#                     # headers and footers (headers: take out if not first page; footers: take out)
#                     if block['is_header']:
#                         if key == 0:
#                             html_div_main = html_div_main + '<p style=display: inline-block;margin-left: {tag_para_left}px;>' + html_span + '</p>' + '\n'
#                         else:
#                             html_div_header = html_div_header + '<p>' + html_span + '</p>' + '\n'
#                     elif block['is_footer']:
#                         html_div_footer = html_div_footer + '<p>' + html_span + '</p>' + '\n'
#                     else:
#                         # main div
#                         if block['is_title']:
#                             html_div_main = html_div_main + '<h3 style=margin-left: {tag_para_left}px;>' + html_span + '</h1>' + '\n'
#                         else:
#                             html_div_main = html_div_main + '<p style=display: inline-block;margin-left: {tag_para_left}px;>' + html_span + '</p>' + '\n'

#             elif block['is_footnote']:
#                 for k, para in enumerate(block['lines']):
#                     html_span = ''
#                     for l, span in enumerate(block['lines'][k]['spans']):
#                         # mandatory tags
#                         tag_span_fontsize = str(round(span['size'], 2))
#                         tag_span_left = str(round(span['bbox'][0], 2))

#                         # beginning and endding tags for span
#                         # tag_span_begin = f'<span id="span_{i}_{j}_{k}_{l}" style="font-size: {tag_span_fontsize}px; margin-left: {tag_span_left}px;">'
#                         tag_span_begin = f'<span id="span_{i}_{j}_{k}_{l}">'
#                         tag_span_end = '</span>'

#                         # optional tags
#                         if span['is_superscript']:
#                             tag_span_begin = tag_span_begin + '<sup>'
#                             tag_span_end = '</sup>' + tag_span_end
#                         if span['is_italic']:
#                             tag_span_begin = tag_span_begin + '<i>'
#                             tag_span_end = '</i>' + tag_span_end
#                         if span['is_bold']:
#                             tag_span_begin = tag_span_begin + '<b>'
#                             tag_span_end = '</b>' + tag_span_end

#                         # string for span
#                         html_span = html_span + tag_span_begin + span['text'] + tag_span_end

#                     html_div_footnote = html_div_footnote + '<p>' + html_span + '</p>' + '\n'

#         if block['type'] == 1: # image block -- treat as paragraph
#             tag_image = f"""
#             <img id="img_{i}_{j}"
#             src="data:image/{block['ext']};base64,{base64.b64encode(block['image']).decode('utf-8')}"
#             style="display: inline-block;margin-left: {block['bbox'][0]}px;
#             width: {str(block['bbox'][2]-block['bbox'][0])}px;
#             height: {str(block['bbox'][3]-block['bbox'][1])}px;">
#             """
#             html_div_main = html_div_main + '<p>' + tag_image + '</p>' + '\n'

# html_divs = '<div>' + html_div_main + '</div>' + '<hr>' + '<div>' + html_div_footnote + '</div>'

# title = ''
# style = ""
# html_string = produce_html(divs=html_divs, title=title, style=style)



# with open('aa.html', 'w', encoding='utf-8') as f:
#     f.write(html_string)


# API calls
# # ------------------------------- Python -------------------------------
# import requests
# import base64

# url = "http://localhost:8000/convert_pdf"
# files = {"file": open("DATA/training_data/leg_trend/Issue_pdf/201711_phy_평석_02-형사-주거침입절도의 경우에는 통신사실확인자료를 요청할 수 없다.pdf", "rb")}
# response = requests.post(url, files=files)

# if response.status_code == 200:
#     result = response.json()
#     result = eval(result['result'])
# else:
#     print("Error:", response.status_code, response.text)

# # convert image strings to bytstring
# for k, v in result.items():
#     for idx, el in enumerate(v):
#         if v[idx]['tag'] == '<img>':
#             v[idx]['line'] = base64.b64decode(v[idx]['line'])


# # -------------------------------- CURL --------------------------------
# curl -X POST -F "file=@../DATA/training_data/Issue_pdf/201711_phy_평석_02-형사-주거침입절도의 경우에는 통신사실확인자료를 요청할 수 없다.pdf" http://localhost:8000/convert_pdf


# # -------------------------------- C# --------------------------------
# using System;
# using System.Net.Http;
# using System.IO;
# using System.Collections.Generic;
# using System.Text;
# using Newtonsoft.Json.Linq;

# class Program
# {
#     static void Main(string[] args)
#     {
#         var url = "http://localhost:8000/convert_pdf";
#         var filePath = @"DATA/training_data/leg_trend/Issue_pdf/201711_phy_평석_02-형사-주거침입절도의 경우에는 통신사실확인자료를 요청할 수 없다.pdf";

#         using (var client = new HttpClient())
#         {
#             using (var formData = new MultipartFormDataContent())
#             {
#                 using (var fileStream = File.OpenRead(filePath))
#                 {
#                     var fileName = Path.GetFileName(filePath);
#                     formData.Add(new StreamContent(fileStream), "file", fileName);

#                     var response = client.PostAsync(url, formData).Result;

#                     if (response.IsSuccessStatusCode)
#                     {
#                         var result = JObject.Parse(response.Content.ReadAsStringAsync().Result)["result"].ToString();
#                         var resultDict = JObject.Parse(result).ToObject<Dictionary<string, List<Dictionary<string, string>>>>();

#                         // convert image strings to bytstring
#                         foreach (var kvp in resultDict)
#                         {
#                             foreach (var item in kvp.Value)
#                             {
#                                 if (item["tag"] == "<img>")
#                                 {
#                                     item["line"] = Convert.ToBase64String(Encoding.UTF8.GetBytes(item["line"]));
#                                 }
#                             }
#                         }
#                     }
#                     else
#                     {
#                         Console.WriteLine("Error: " + response.StatusCode + " " + response.Content.ReadAsStringAsync().Result);
#                     }
#                 }
#             }
#         }
#     }
# }

# %% call function with pdf doc as bytes
# with open('../DATA/training_data/leg_trend/Issue_pdf/202011_phy_평석_04-전동 퀵보드 이용자의 음주운전 처벌-형사-수정.pdf', 'rb') as f:
with open('DATA/training_data/leg_trend/Issue_pdf/201711_phy_평석_02-형사-주거침입절도의 경우에는 통신사실확인자료를 요청할 수 없다.pdf', 'rb') as f:
    pdf_bytes = f.read()
    result = convert_pdf(pdf_bytes)
# %%
