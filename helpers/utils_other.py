

from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib import patches
from paddleocr import PPStructure,draw_structure_result
from sklearn.cluster import KMeans
import numpy as np
import numbers

# %% bbox-related utils (applies to: PP, pymupdf, ...)
# functon to estimate the number of columns
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


# function to get area of a bbox
def bbox_area(bbox):
    x0, y0, x1, y1 = bbox
    return (x1 - x0) * (y1 - y0)


# function to get intersection area of two bboxes
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



# function to calculate overlap percentage of two bboxes
def bbox_overlap_percentage(
    bbox1 = None,  # list or tuple (x_min, y_min, x_max, y_max) if comparison_option == 'bbox_v ...'
                   # othewise nested list or tuple
    bbox2 = None,  # list or tuple (x_min, y_min, x_max, y_max) if comparison_option == '..._v bbox'
                   # nested list or tuple if comparison_option == '..._v_list'
                   # None if comparison_option == 'list_v_self'
    comparison_option = None, #'bbox_v_list', # 'bbox_v_bbox', 'list_v_list', 'list_self'
    threshold = 0.5  # threshold for overlap percentage
):

    # bbox1 = [100, 100, 200, 200]
    # bbox2 = [150, 150, 250, 250]
    # bbox1 = [[160, 92, 443, 121], [100, 164, 510, 174], [100, 100, 250, 250]]
    # bbox2 = [[160, 92, 443, 121], [266, 164, 510, 174], [150, 150, 250, 250]]
    # bbox2 = [[160, 92, 443, 121], [266, 164, 510, 174], [266, 164, 510, 174]]
    # threshold = 0.5

    if comparison_option == 'bbox_v_bbox':

        # Convert bounding boxes to NumPy arrays
        bbox1 = np.array(bbox1)
        bbox2 = np.array(bbox2)

        # type checking
        assert(all([bbox1.ndim == 1, bbox2.ndim == 1]))

        # Calculate the coordinates of the intersection rectangle
        x_min = np.maximum(bbox1[0], bbox2[0])
        y_min = np.maximum(bbox1[1], bbox2[1])
        x_max = np.minimum(bbox1[2], bbox2[2])
        y_max = np.minimum(bbox1[3], bbox2[3])

        # Calculate the area of the intersection rectangle
        intersection_area = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)

        # Calculate the area of each bounding box in the list
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate the percentage of overlapping area relative to the area of each bounding box in the list
        overlap_percentage_bbox1 = intersection_area / bbox1_area
        overlap_percentage_bbox2 = intersection_area / bbox2_area

        # Get the indices of the intersecting bounding boxes
        intersecting_bbox_indices_bbox1 = np.where(overlap_percentage_bbox1 > threshold)[0]
        intersecting_bbox_indices_bbox2 = np.where(overlap_percentage_bbox2 > threshold)[0]
        intersecting_bbox_indices = np.unique(
            np.concatenate((intersecting_bbox_indices_bbox1, intersecting_bbox_indices_bbox2))
            )

        return {'indice': intersecting_bbox_indices,
                'overlap_pct_bbox1': overlap_percentage_bbox1,
                'overlap_pct_bbox2': overlap_percentage_bbox2}

    elif comparison_option == 'bbox_v_list':

        # Convert bounding boxes to NumPy arrays
        bbox = np.array(bbox1)
        bbox_list = np.array(bbox2)

        # type checking
        assert(all([bbox.ndim == 1, bbox_list.ndim == 2]))

        # Calculate the coordinates of the intersection rectangle
        x_min = np.maximum(bbox[0], bbox_list[:, 0])
        y_min = np.maximum(bbox[1], bbox_list[:, 1])
        x_max = np.minimum(bbox[2], bbox_list[:, 2])
        y_max = np.minimum(bbox[3], bbox_list[:, 3])

        # Calculate the area of the intersection rectangle
        intersection_area = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)

        # Calculate the area of each bounding box in the list
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        bbox_list_area = (bbox_list[:, 2] - bbox_list[:, 0]) * (bbox_list[:, 3] - bbox_list[:, 1])

        # Calculate the percentage of overlapping area relative to the area of each bounding box in the list
        overlap_percentage_bbox = intersection_area / bbox_area
        overlap_percentage_bbox_list = intersection_area / bbox_list_area

        # Get the indices of the intersecting bounding boxes
        intersecting_bbox_indices_bbox = np.where(overlap_percentage_bbox > threshold)[0]
        intersecting_bbox_indices_bbox_list = np.where(overlap_percentage_bbox_list > threshold)[0]
        intersecting_bbox_indices = np.unique(
            np.concatenate((intersecting_bbox_indices_bbox, intersecting_bbox_indices_bbox_list))
            )

        return {'indice': intersecting_bbox_indices,
                'overlap_pct_bbox1': overlap_percentage_bbox[intersecting_bbox_indices],
                'overlap_pct_bbox2': overlap_percentage_bbox_list[intersecting_bbox_indices]}

    elif comparison_option == 'list_v_list':
        pass

        # Convert bounding boxes to NumPy arrays
        bbox_list1 = np.array(bbox1)
        bbox_list2 = np.array(bbox2)

        # type checking
        assert(all([bbox_list1.ndim == 2, bbox_list2.ndim == 2]))

        # calcular coordicates of all possible intersection rectangles
        x_min = np.maximum(bbox_list1[:, 0][:, np.newaxis], bbox_list2[:, 0])
        y_min = np.maximum(bbox_list1[:, 1][:, np.newaxis], bbox_list2[:, 1])
        x_max = np.minimum(bbox_list1[:, 2][:, np.newaxis], bbox_list2[:, 2])
        y_max = np.minimum(bbox_list1[:, 3][:, np.newaxis], bbox_list2[:, 3])

        # Calculate the area of the intersection rectangle
        intersection_area = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)

        # Calculate the area of each bounding box in the list
        bbox_list1_area = (bbox_list1[:, 2] - bbox_list1[:, 0]) * (bbox_list1[:, 3] - bbox_list1[:, 1])
        bbox_list2_area = (bbox_list2[:, 2] - bbox_list2[:, 0]) * (bbox_list2[:, 3] - bbox_list2[:, 1])

        # Calculate the percentage of overlapping area relative to the area of each bounding box in the list
        overlap_percentage_bbox_list1 = intersection_area / bbox_list1_area[:, np.newaxis]
        overlap_percentage_bbox_list2 = intersection_area / bbox_list2_area[:, np.newaxis]

        # Get the indices of the intersecting bounding boxes
        overlap_indice_bbox_list1 = np.argwhere(overlap_percentage_bbox_list1 > threshold)
        # README np.argwhere outputs list of [row, col]; np.where outputs list of row and list of col
        overlap_indice_bbox_list1 = [tuple(el) for el in overlap_indice_bbox_list1]  # convert to list of tuples

        overlap_indice_bbox_list2 = np.argwhere(overlap_percentage_bbox_list2 > threshold)
        overlap_indice_bbox_list2 = [tuple(el) for el in overlap_indice_bbox_list2]  # convert to list of tuples

        # remove identical pairs & concatenate
        overlap_indice_bbox_list1 = list(set([tuple(sorted(el)) for el in overlap_indice_bbox_list1]))
        overlap_indice_bbox_list2 = list(set([tuple(sorted(el)) for el in overlap_indice_bbox_list2]))
        overlap_indice = list(set(overlap_indice_bbox_list1 + overlap_indice_bbox_list2))

        # get overlapping percentages in lists
        overlap_percentage_bbox1 = [overlap_percentage_bbox_list1[el[0], el[1]] for el in overlap_indice_bbox_list1]
        overlap_percentage_bbox2 = [overlap_percentage_bbox_list2[el[0], el[1]] for el in overlap_indice_bbox_list2]

        return {'indice': overlap_indice,
                'overlap_pct_bbox1': overlap_percentage_bbox1,
                'overlap_pct_bbox2': overlap_percentage_bbox2}

    elif comparison_option == 'list_self':

        # Convert bounding boxes to NumPy arrays
        bbox_list = np.array(bbox1)

        # type checking
        assert(all([bbox_list.ndim == 2, bbox2 is None]))

        # Calculate the coordinates of the intersection rectangle for each pair of bounding boxes
        x_min = np.maximum(bbox_list[:, 0][:, np.newaxis], bbox_list[:, 0])
        y_min = np.maximum(bbox_list[:, 1][:, np.newaxis], bbox_list[:, 1])
        x_max = np.minimum(bbox_list[:, 2][:, np.newaxis], bbox_list[:, 2])
        y_max = np.minimum(bbox_list[:, 3][:, np.newaxis], bbox_list[:, 3])

        # Calculate the area of the intersection rectangle for each pair of bounding boxes
        intersection_area = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)

        # Calculate the area of each bounding box in the list
        bbox_area = (bbox_list[:, 2] - bbox_list[:, 0]) * (bbox_list[:, 3] - bbox_list[:, 1])

        # Set the diagonal elements of the intersection area matrix to zero to avoid self-intersection
        np.fill_diagonal(intersection_area, 0)

        # get overlapping percentages
        overlap_percentage = intersection_area / bbox_area[:, np.newaxis]

        # get indices of overlapping bboxes
        overlap_indice = np.argwhere(overlap_percentage > threshold)  # np.argwhere outputs list of [row, col]
                                                                      # np.where outputs list of row and list of col
        overlap_indice = [tuple(el) for el in overlap_indice]  # convert to list of tuples

        # remove identical pairs
        # overlap_indice = list(set([tuple(sorted(el)) for el in overlap_indice]))

        # get overlapping percentages in list
        overlap_percentage = [overlap_percentage[el[0], el[1]] for el in overlap_indice]

        return {'indice': overlap_indice,
                'overlap_pct_bbox1': overlap_percentage,
                'overlap_pct_bbox2': None}

    else:
        raise ValueError('Unknown comparison option')


# function to render multiple groups of bboxes with an image, with matpliotlib
def render_bboxes_2_with_image(
    image_byte: str,
    bbox_groups: list, # list of list of bboxes in [x0, y0, x1, y1] format
    labels: list,
    label_index: int, # index of bbox group to label
    color: list,
    alpha=0.5, thickness=1, dpi=200, fontsize=6, display_labels=True):

    img = Image.open(BytesIO(image_byte))
    height, width = img.size
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off') # hide axis

    # add bboxes
    for bboxes, c in zip(bbox_groups, color):
        for bbox in bboxes:
            x0, y0, x1, y1 = bbox
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=thickness, edgecolor=c, facecolor='none', alpha=alpha)
            ax.add_patch(rect)

    # add labels
    if display_labels:
        for idx, bboxes in enumerate(bbox_groups[label_index]):
            x0, y0, x1, y1 = bboxes
            ax.text(x0, y0, labels[idx], fontsize=fontsize, color='green')#, bbox=dict(facecolor='white', alpha=0.5))

    plt.imshow(img)


# %% functino to parse layout
def parse_layout(
    parser: str,
    image_byte: str,
    pp_font_path: str = None, # for PP
    use_gpu: bool = False # for PP
    ):

    if parser == 'PP':
        layout_engine = PPStructure(
            ocr=False,
            show_log=False,
            use_gpu=use_gpu,
            image_orientation=False,
            layout=True,
            table=True,
            # ocr_order_method=None,
            # mode='structure',
            # recovery=Flase,
            # use_pdf2docx_api=False,
            # lang='ch',
            # det=True,
            # rec=True,
            # type='ocr',
            # ocr_version='PP-OCRv3',
            # structure_version='PP-StructureV2'
            # =====================
            # use_xpu=False,
            # use_npu=False,
            # ir_optim=True,
            # use_tensorrt=False,
            # min_subgraph_size=15,
            # precision='fp32',
            # gpu_mem=500,
            # image_dir=None,
            # page_num=0,
            # det_algorithm='DB',
            # det_model_dir='/home/lstm/.paddleocr/whl/det/ch/ch_PP-OCRv3_det_infer',
            # det_limit_side_len=960,
            # det_limit_type='max',
            # det_box_type='quad',
            # det_db_thresh=0.3,
            # det_db_box_thresh=0.6,
            # det_db_unclip_ratio=1.5,
            # max_batch_size=10,
            # use_dilation=False,
            # det_db_score_mode='fast',
            # det_east_score_thresh=0.8,
            # det_east_cover_thresh=0.1,
            # det_east_nms_thresh=0.2,
            # det_sast_score_thresh=0.5,
            # det_sast_nms_thresh=0.2,
            # det_pse_thresh=0,
            # det_pse_box_thresh=0.85,
            # det_pse_min_area=16,
            # det_pse_scale=1,
            # scales=[8, 16, 32],
            # alpha=1.0,
            # beta=1.0,
            # fourier_degree=5,
            # rec_algorithm='SVTR_LCNet',
            # rec_model_dir='/home/lstm/.paddleocr/whl/rec/ch/ch_PP-OCRv3_rec_infer',
            # rec_image_inverse=True,
            # rec_image_shape='3, 48, 320',
            # rec_batch_num=6,
            # max_text_length=25,
            # rec_char_dict_path='/home/lstm/miniconda3/envs/pdf2html/lib/python3.9/site-packages/paddleocr/ppocr/utils/ppocr_keys_v1.txt',
            # use_space_char=True,
            # vis_font_path='./doc/fonts/simfang.ttf',
            # drop_score=0.5,
            # e2e_algorithm='PGNet',
            # e2e_model_dir=None,
            # e2e_limit_side_len=768,
            # e2e_limit_type='max',
            # e2e_pgnet_score_thresh=0.5,
            # e2e_char_dict_path='./ppocr/utils/ic15_dict.txt',
            # e2e_pgnet_valid_set='totaltext',
            # e2e_pgnet_mode='fast',
            # use_angle_cls=False,
            # cls_model_dir=None,
            # cls_image_shape='3, 48, 192',
            # label_list=['0', '180'],
            # cls_batch_num=6,
            # cls_thresh=0.9,
            # enable_mkldnn=False,
            # cpu_threads=10,
            # use_pdserving=False,
            # warmup=False,
            # sr_model_dir=None,
            # sr_image_shape='3, 32, 128',
            # sr_batch_num=1,
            # draw_img_save_dir='./inference_results',
            # save_crop_res=False,
            # crop_res_save_dir='./output',
            # use_mp=False,
            # total_process_num=1,
            # process_id=0,
            # benchmark=False,
            # save_log_path='./log_output/',
            # use_onnx=False,
            # output='./output',
            # table_max_len=488,
            # table_algorithm='TableAttn',
            # table_model_dir='/home/lstm/.paddleocr/whl/table/ch_ppstructure_mobile_v2.0_SLANet_infer',
            # merge_no_span_structure=True,
            # table_char_dict_path='/home/lstm/miniconda3/envs/pdf2html/lib/python3.9/site-packages/paddleocr/ppocr/utils/dict/table_structure_dict_ch.txt',
            # layout_model_dir='/home/lstm/.paddleocr/whl/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer',
            # layout_dict_path='/home/lstm/miniconda3/envs/pdf2html/lib/python3.9/site-packages/paddleocr/ppocr/utils/dict/layout_dict/layout_cdla_dict.txt',
            # layout_score_threshold=0.5,
            # layout_nms_threshold=0.5,
            # kie_algorithm='LayoutXLM',
            # ser_model_dir=None,
            # re_model_dir=None,
            # use_visual_backbone=True,
            # ser_dict_path='../train_data/XFUND/class_list_xfun.txt',
            )

        # get layout with PP and sort by location
        result = layout_engine(image_byte)
        result = sorted(result, key=lambda x: x['bbox'][1])

        # convert the bytes object to a PIL image
        img = Image.open(BytesIO(image_byte))
        layout_image_array = draw_structure_result(img, result, font_path=pp_font_path)
        result = [{key: value for key, value in el.items() if key != 'img'} for el in result]

    return result, layout_image_array


# %% function to parse texts (page)
def parse_texts_page(
    parser: str,
    pymupdf_page_object: None,
    image_byte_for_ocr: str, # byte string
    parse_option: str = 'dict'
):
    if parser == 'pymupdf':
        blocks = pymupdf_page_object.get_text(parse_option)

    elif parser == 'naver':
        pass

    return blocks


# function to parse texts (area)
def parse_texts_area(
    parser: str,
    pymupdf_page_object: None,
    area_label: list, # ["reference", "header", "title", ...]
    parse_option: str = 'dict'
):
    if parser == 'pymupdf':

        def make_text(words):
            """Return textstring output of get_text("words").

            Word items are sorted for reading sequence left to right,
            top to bottom.
            """
            line_dict = {}  # key: vertical coordinate, value: list of words
            words.sort(key=lambda w: w[0])  # sort by horizontal coordinate
            for w in words:  # fill the line dictionary
                y1 = round(w[3], 1)  # bottom of a word: don't be too picky!
                word = w[4]  # the text of the word
                line = line_dict.get(y1, [])  # read current line content
                line.append(word)  # append new word
                line_dict[y1] = line  # write back to dict
            lines = list(line_dict.items())
            lines.sort()  # sort vertically
            return "\n".join([" ".join(line[1]) for line in lines])

        page = pymupdf_page_object

        """
        -------------------------------------------------------------------------------
        Identify the rectangle.
        -------------------------------------------------------------------------------
        """
        rect = page.first_annot.rect  # this annot has been prepared for us!
        # Now we have the rectangle ---------------------------------------------------

        """
        Get all words on page in a list of lists. Each word is represented by:
        [x0, y0, x1, y1, word, bno, lno, wno]
        The first 4 entries are the word's rectangle coordinates, the last 3 are just
        technical info (block number, line number, word number).
        The term 'word' here stands for any string without space.
        """

        words = page.get_text("words")  # list of words on page

        """
        We will subselect from this list, demonstrating two alternatives:
        (1) only words inside above rectangle
        (2) only words insertecting the rectangle

        The resulting sublist is then converted to a string by calling above funtion.
        """

        # ----------------------------------------------------------------------------
        # Case 1: select the words *fully contained* in the rect
        # ----------------------------------------------------------------------------
        mywords = [w for w in words if fitz.Rect(w[:4]) in rect]

        print("Select the words strictly contained in rectangle")
        print("------------------------------------------------")
        print(make_text(mywords))

        # ----------------------------------------------------------------------------
        # Case 2: select the words *intersecting* the rect
        # ----------------------------------------------------------------------------
        mywords = [w for w in words if fitz.Rect(w[:4]).intersects(rect)]

        print("\nSelect the words intersecting the rectangle")
        print("-------------------------------------------")
        print(make_text(mywords))





    elif parser == 'naver':
        pass

    return blocks

# function to create html from div dtring
# %% html-related utils
def produce_html(divs, title, style):

    html_string = f"""
        <html>
        <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
        {style}
        </style>
        </head>
        <body>
        {divs}
        </body>
        </html>
        """

    return html_string


# %% other utils
def check_bits(n):
    bit_0 = n & 0b00001 != 0
    bit_1 = n & 0b00010 != 0
    bit_2 = n & 0b00100 != 0
    bit_3 = n & 0b01000 != 0
    bit_4 = n & 0b10000 != 0

    return bit_0, bit_1, bit_2, bit_3, bit_4
