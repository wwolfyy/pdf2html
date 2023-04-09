# import os
# print(os.getcwd())
# if os.getcwd().endswith('UI/pages'):
#     os.chdir('../')
#     print(os.getcwd())
# if os.getcwd().lower().endswith('pdf2html'):
#     os.chdir('UI')    
#     print(os.getcwd())

# import sys
#[... more imports ...]
# sys.path.append('./helpers/')

import streamlit as st
# from stutils import load_css
from PIL import Image, ImageOps
import fitz
from stutils import detect_text_gcv, produce_html
import io
import base64
# fitz.__doc__


class PDFParser:

    # pagetitle = 'PDF Parser'

    @st.cache_data    
    def view(_self, _doc, dimension, pagenum):  

        # doc = fitz.open('부산지법_2018나55364_판결서.pdf', filetype='pdf')
        # pagenum = 1

        # doc = fitz.open(stream=file.read(), filetype='pdf')
        # dimension = (doc[0].get_pixmap().irect[2], doc[0].get_pixmap().irect[3])

        # c1, c2, c3 = st.columns(3)
        c11, c12 = st.columns(2)    
        
        # with c1:
            # pagenum = st.selectbox('select page', (range(1, len(doc)+1)))
         

        with c11:

            image = _doc[pagenum-1].get_pixmap().tobytes('png')                
            # image = ImageOps.expand(image, border=20, fill='red')
            
            st.image(image, width=dimension[0])
            # save bytesio image for HTML rendering
            with open('tmp_image.png', 'wb') as f:
                f.write(image)

            # send api call for OCR
            ocrtext = detect_text_gcv(image)                

        with c12:

            # draw rectabgles on the html file, with id assigned to each rectangle

            html_divs = ''
            for i, text in enumerate(ocrtext):

                if i == 0:
                    continue
                
                if text.description!=None:
                    
                    rect = ([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])

                    html_divs = html_divs + (f"""
                    <div id="rect_{i}" style="position: absolute; 
                    left: {rect[0][0]}px; top: {rect[0][1]}px; 
                    width: {rect[1][0]-rect[0][0]}px; 
                    height: {rect[2][1]-rect[0][1]}px; 
                    border: 1px solid red; 
                    background-color: rgba(255, 0, 0, 0.3);"
                    title="{text.description}"></div>
                    ></div>
                    """)
            
                            
            b64image = base64.b64encode(image).decode('utf-8')

            html_divs = f"""
            <div><img src="data:image/png;base64,{b64image}" style="position: absolute; 
            left: 0px; top: 0px;" 
            alt="image"></div>""" + html_divs



            imgpath = "file:///home/lstm/Github/pdf2html/tmp_image.png"

            # produce html file
            html_string = produce_html('tmp.html', imgpath, html_divs)
            
            # display html file

            # HtmlFile = open('tmp.html', 'r', encoding='utf-8')
            # source_code = HtmlFile.read()  
            st.components.v1.html(html_string, width=dimension[0], height=dimension[1], scrolling=False)


        # st.write(html_string)
