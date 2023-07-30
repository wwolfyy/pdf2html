import os, sys
# import cv2
# import fitz
from paddleocr import PPStructure,draw_structure_result,save_structure_res
from PIL import Image
import matplotlib.pyplot as plt

# function to detect PP layout, save result, and optionally draw bbox
def detect_layout_pp(
        image,
        save_result: bool = False,
        savefolder: str = './output',
        savename: str = None,
        draw_bbox: bool = False,
        font_path: str = '/home/lstm/Github/pdf2html/PP/NotoSansKR-Regular.otf'
        ):

    # get layout
    table_engine = PPStructure(show_log=True) #, image_orientation=True)
    result = table_engine(image)

    if save_result:
        save_structure_res(result, savefolder, os.path.basename(savename).split('.')[0])

    if draw_bbox:
        im_show = draw_structure_result(image, result,font_path=font_path)
        im_show = Image.fromarray(im_show)

        return result, im_show
    else:
        return result


# function to render PP layout from image array
def render_PP_layout_from_array(image_array):
    im_show = Image.fromarray(image_array)
    im_show.show()

    dpi = 200
    height, width = im_show.size
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off') # hide axis
    plt.imshow(im_show)
