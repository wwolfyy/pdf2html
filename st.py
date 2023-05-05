import streamlit as st
from stutils import load_css, create_html_from_byteio
from streamlit_option_menu import option_menu
import streamlit_javascript as st_js
import fitz, cv2
import os, sys
from paddleocr import PPStructure #,draw_structure_result,save_structure_res
import numpy as np
from io import BytesIO
import re

# add path
sys.path.append(os.path.abspath('./helpers'))

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

from helpers.utils_fitz import get_text_length, get_num_images, is_scanned_pdf, \
    pdf2image, parse_pdf
    #get_block_bbox, parse_pdf, get_fonts_meta, pdf2image

from PIL import Image


if 'dimensions' not in st.session_state:
    st.session_state['dimensions'] = None

if 'parsed_json' not in st.session_state:
    st.session_state['parsed_json'] = None

if 'title_blocks_list' not in st.session_state:
    st.session_state['title_blocks_list'] = []

if 'parsing_done' not in st.session_state:
    st.session_state['parsing_done'] = False

if 'query_embedding' not in st.session_state:
    st.session_state['query_embedding'] = None

if 'client' not in st.session_state:
    st.session_state['client'] = None

if 'doc' not in st.session_state:
    st.session_state['doc'] = None

if 'end_reached' not in st.session_state:
    st.session_state['end_reached'] = False

print(os.getcwd())

st.set_page_config(
    page_title="DeepParser",
    page_icon="arrows_clockwise",
    layout="wide",
)

load_css()


from sklearn.cluster import KMeans

def estimate_columns(x_coords, max_columns=2):

    # Try different numbers of clusters (columns) and find the best fit using the k-means algorithm
    best_inertia = float('inf')
    best_num_columns = 1

    for num_columns in range(1, max_columns + 1):
        kmeans = KMeans(n_clusters=num_columns, random_state=0).fit(x_coords)
        inertia = kmeans.inertia_

        if inertia < best_inertia:
            best_inertia = inertia
            best_num_columns = num_columns

    return best_num_columns, kmeans.fit(x_coords).labels_


def bbox_area(bbox):
    x0, y0, x1, y1 = bbox
    return (x1 - x0) * (y1 - y0)


def bbox_intersection(bbox1, bbox2):
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2

    x0_int = max(x0_1, x0_2)
    y0_int = max(y0_1, y0_2)
    x1_int = min(x1_1, x1_2)
    y1_int = min(y1_1, y1_2)

    if x0_int < x1_int and y0_int < y1_int:
        return (x1_int - x0_int) * (y1_int - y0_int)
    else:
        return 0


def bbox_overlap_percentage(bbox1, bbox2):
    intersection = bbox_intersection(bbox1, bbox2)

    if intersection == 0:
        return 0, 0

    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)

    return intersection / area1 * 100, intersection / area2 * 100


choose = option_menu("Deep Neural Network Document & Image Parser", ["PDF: 유사 단락 검색 모델"], # & 이미지"],
                        icons=['file-earmark-pdf-fill'],
                        menu_icon="pencil-square", default_index=0,
                        orientation='horizontal',
                        styles={
    # "container": {"padding": "5!important", "background-color": "#fafafa"},
    "icon": {"color": "orange", "font-size": "25px"},
    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "skyblue"},
    }
)

if choose == "PDF: 유사 단락 검색 모델":

    st.write('검색 대상 문서을 아래에 업로드하고 검색어를 입력하면 검색어와 가장 연관성 높은 섹션을 찾아줍니다.')

    st.markdown(
        """
        <p style='font-size:12px;'>
        *이 페이지는 일반에 공개되고 있습니다. <br>
        *복합적으로 작동하는 복수의 문서 인식 모델들이 낮은 사양의 서버에서 운영되고 있어 문서 인식에 장당
        10~25초 정도의 시간이 소요됩니다. <br>
        *또한 정확성보다 속도에 촛점을 둔 임베딩 모델이 사용되기 때문에 정확성이 떨어질 수 있습니다.<p>
        """,
        unsafe_allow_html=True
    )

    file = st.file_uploader('upload pdf', type=['pdf']) #, 'jpg', 'png', 'jpeg'])

    if not st.session_state['parsing_done']:
        if file is not None:

            # determine the file type
            if file.name.endswith('.pdf'):
                filetype = 'pdf'
            elif file.name.endswith('.jpg') or file.name.endswith('.png') or file.name.endswith('.jpeg'):
                filetype = 'image'

            # doc = fitz.open('부산지법_2018나55364_판결서.pdf', filetype='pdf')
            # pagenum = 1
            doc = fitz.open(stream=file.read(), filetype='pdf')
            rotation = doc.load_page(0).rotation
            if not rotation == 0:
                st.warning('테스트 서버에서는 세로 형식의 문서만 지원합니다.')
                st.experimental_rerun()
            dimension = (doc[0].get_pixmap().irect[2], doc[0].get_pixmap().irect[3])

            st.session_state['doc'] = doc
            st.session_state['dimension'] = dimension

            # parse structure with PP
            table_engine = PPStructure(show_log=True) #, image_orientation=True)

            with st.spinner('문서 구조를 인식중입니다'):

                parsed_dict = {}

                # get structure
                for pagenum, page in enumerate(doc):

                    page_image = page.get_pixmap()

                    # convert to np array
                    pic = np.array(Image.open(BytesIO(page_image.tobytes())))

                    # get layout
                    result = table_engine(pic)

                    xcoords = [box['bbox'][0] for box in result]

                    if len(xcoords) > 1:
                        num_cols, grouping = estimate_columns(np.array(xcoords).reshape(-1, 1))
                    else:
                        num_cols = 1
                        grouping = [0]

                    # group xcoords by column, then sort by y0, if more than 1 column
                    if num_cols > 1:
                        col1 = [xcoords[i] for i in range(len(xcoords)) if grouping[i] == 0]
                        col2 = [xcoords[i] for i in range(len(xcoords)) if grouping[i] == 1]

                        group1 = [box for box in result if box['bbox'][0] in col1]
                        group2 = [box for box in result if box['bbox'][0] in col2]

                        group1 = sorted(group1, key=lambda x: x['bbox'][1])
                        group2 = sorted(group2, key=lambda x: x['bbox'][1])

                        # tag first element of group2 as "continued" if continued from group1
                        if group1[-1]['type'] == 'text' and group2[0]['type'] == 'text':
                            group2[0]['continued'] = True

                        result = group1 + group2

                    else:
                        result = sorted(result, key=lambda x: x['bbox'][1])

                    # get texts using pymupdf
                    page_blocks = page.get_text('blocks')

                    # put blocks into result
                    for i, box in enumerate(result):

                        bbox1 = box['bbox']
                        result[i]['text_blocs'] = []

                        for block in page_blocks:

                            block_bbox = block[:4]
                            # block_text = block[4]

                            bbox2 = block_bbox

                            # determine which block in result overlaps more than 50% with the block in page_blocks
                            overlap_structure, overlap_pdf = bbox_overlap_percentage(bbox1, bbox2)

                            if overlap_structure > 50 and overlap_pdf > 50:
                                result[i]['text_blocs'] = result[i]['text_blocs'] + [block]

                    parsed_dict[pagenum] = result

                # mark "continued" for continued text blocks
                for key in parsed_dict.keys():
                    if key != 0:
                        for box in parsed_dict[key]:
                            if box['type'] == 'text':
                                dictkey = key - 1
                                if parsed_dict[dictkey][-1]['type'] == 'text':
                                    box['continued'] = True

                st.session_state['parsed_json'] = parsed_dict

                # extract title blocks from parsed_dict, where parsed_dict[key][i]['type'] == 'title'
                title_blocks = {}
                for key in parsed_dict.keys():
                    title_blocks[key] = [box for box in parsed_dict[key] if box['type'] == 'title']

                # extract each title block, and append bbox, text, and page number
                title_blocks_list = []
                for key in title_blocks.keys():
                    if len(title_blocks[key]) > 0:
                        for box in title_blocks[key]:
                            # remove non-chareter symbols
                            if len(box['text_blocs']) > 0:
                                fixed_text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', box['text_blocs'][0][4])
                            else:
                                fixed_text = ''
                            # fix punctuations etc.
                            # fixed_text = fix_punct(box['text_blocs'][0][4])
                            title_blocks_list.append([box['bbox'], fixed_text, key])

                # get embedding for title block texts
                import nlpcloud
                title_token_list = [bloc[1] for bloc in title_blocks_list]

                if len(title_token_list) == 0:
                    st.warning('문서 구조 인식에 실패했습니다. 다른 문서를 시도해주세요.')
                    st.stop()
                    # st.experimental_rerun()

                client = nlpcloud.Client(
                    "paraphrase-multilingual-mpnet-base-v2",
                    "ba1502dcfefb24cea5e6abf80a6be1d5c755beba"
                    )
                st.session_state['client'] = client

                title_embeddings = client.embeddings(title_token_list)

                # zip with title_blocks_list
                title_blocks_list = [title_blocks_list[i] + [title_embeddings['embeddings'][i]] for i in range(len(title_blocks_list))]

                st.session_state['title_blocks_list'] = title_blocks_list

                parsing_done = True
                st.session_state['parsing_done'] = parsing_done

                st.success('문서 구조 인식이 완료되었습니다.')

    if st.session_state['parsing_done']:

        with st.form('query_form'):
            query = st.text_input('아래에 찾고자하는 단락의 내용을 입력하세요. 가장 유사한 단락이 표시됩니다.')
            submit_query = st.form_submit_button('검색')

        if submit_query:
            st.session_state['query'] = query

            # get embedding for query
            if query != '':
                client = st.session_state['client']
                query_embedding = client.embeddings([query])
                st.session_state['query_embedding'] = query_embedding

            # calculate cosine similarity between query and title blocks
            from sklearn.metrics.pairwise import cosine_similarity
            if query != '' and query_embedding != {}:
                similarity_scores = cosine_similarity(
                    st.session_state['query_embedding']['embeddings'],
                    [bloc[3] for bloc in st.session_state['title_blocks_list']]
                    )
                similarity_scores = [score[0] for score in similarity_scores]
                st.session_state['similarity_scores'] = similarity_scores

                # get st.session_state['title_block_list'] for top match
                top1 = np.argmax(similarity_scores)
                st.session_state['top1'] = st.session_state['title_blocks_list'][top1]

                # get bbox of top 1 match
                top1_bbox = st.session_state['title_blocks_list'][top1][0]
                st.session_state['top1_bbox'] = top1_bbox

                # get page number of top 1 match
                top1_pagenum = st.session_state['title_blocks_list'][top1][2]
                st.session_state['top1_pagenum'] = top1_pagenum

                # get text of top 1 match
                top1_text = st.session_state['title_blocks_list'][top1][1]
                st.session_state['top1_text'] = top1_text

                # get similarity score of top 1 match
                top1_similarity = similarity_scores[top1]
                st.session_state['top1_similarity'] = top1_similarity



                # # draw bbox of top 1 match on the page where the top 1 match is located
                # page = st.session_state['doc'].load_page(top1_pagenum)
                # page.draw_rect(top1_bbox, color=(0, 0, 1), width=2, overlay=True)

                # # display the page
                # st.image(Image.open(BytesIO(page.get_pixmap()), use_column_width=False))



                # st.write('검색어와 가장 유사한 단락은 다음과 같습니다.')
                # st.write('유사도: ', top1_similarity)
                # st.write('페이지 번호: ', top1_pagenum)
                # st.write('단락 내용: ', top1_text)
                # st.write('단락 위치: ', top1_bbox)
                # st.write('단락이 포함된 페이지')
                # st.pdf_document('matchoutput.pdf', height=500)

                for key in st.session_state['parsed_json'].keys():
                    if st.session_state['end_reached']:
                        break
                    if key < st.session_state['top1_pagenum']:
                        continue
                    else:
                        # get index of top1_bbox
                        for i, box in enumerate(st.session_state['parsed_json'][key]):
                            if box['bbox'] == st.session_state['top1_bbox']:
                                top1_index = i
                                break
                        # get bbox of subsequent blocks
                        subsequent_bboxes = []
                        for box in st.session_state['parsed_json'][key][top1_index+1:]:
                            if box['type'] == 'title':
                                st.session_state['end_reached'] = True
                                break
                            else:
                                subsequent_bboxes = subsequent_bboxes + box['text_blocs']

                if len(subsequent_bboxes) > 0:
                    max_page = subsequent_bboxes[-1][6]
                    pages_to_show = []
                    for pagenum, page in enumerate(st.session_state['doc']):
                        if pagenum < st.session_state['top1_pagenum']:
                            continue
                        elif pagenum > max_page:
                            break
                        else:
                            # get page image
                            page_image = page.get_pixmap()

                            # convert to np array
                            pic = np.array(Image.open(BytesIO(page_image.tobytes())))

                            if pagenum == st.session_state['top1_pagenum']:
                                page.draw_rect(
                                    st.session_state['top1_bbox'],
                                    color=(1, 0, 0),
                                    width=2,
                                    overlay=True
                                    )
                            for i, box in enumerate(subsequent_bboxes):
                                if box[6] == pagenum:
                                    page.draw_rect(
                                        box[:4],
                                        color=(1, 0, 0),
                                        width=2,
                                        overlay=True
                                        )
                            pages_to_show.append(page)
                else:
                    pages_to_show = [st.session_state['doc'].load_page(st.session_state['top1_pagenum'])]
                    pages_to_show[0].draw_rect(
                        st.session_state['top1_bbox'],
                        color=(1, 0, 0),
                        width=2,
                        overlay=True
                        )
                for page in pages_to_show:
                    st.image(BytesIO(page.get_pixmap().tobytes()), use_column_width=False)
