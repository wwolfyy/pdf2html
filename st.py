import streamlit as st
import json
from stutils import load_css, create_html_from_byteio

from streamlit_option_menu import option_menu
import streamlit_javascript as st_js

import os

from views.PDF import PDFParser
from views.Image import IMAGE
import fitz


print(os.getcwd())

st.set_page_config(
    page_title="CrossConv",
    page_icon="arrows_clockwise",
    layout="wide",
)

load_css()


choose = option_menu("Document Cross Parser", ["PDF", "이미지", "HTML 변환"],
                        icons=['file-earmark-pdf-fill', 'image-fill', 'filetype-html'],
                        menu_icon="pencil-square", default_index=0,
                        orientation='horizontal',
                        styles={
    # "container": {"padding": "5!important", "background-color": "#fafafa"},
    "icon": {"color": "black", "font-size": "25px"}, 
    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "gray"},
}
)

if choose == 'PDF':    
    
    st.markdown(
        """
        <h3 style='text-align: center;'><br>PDF OCR Parser</h3>
    
        <p style='text-align: center;'>PDF에서 텍스트, 이미표, 표, 각주 및 레이아웃을 추출합니다.<p>
        """,
        unsafe_allow_html=True                
    )

    # st.title('PDF Parser')   

    file = st.file_uploader('upload pdf', type=['pdf'])

    if file is not None:

        # doc = fitz.open('부산지법_2018나55364_판결서.pdf', filetype='pdf')
        # pagenum = 1
        doc = fitz.open(stream=file.read(), filetype='pdf')
        dimension = (doc[0].get_pixmap().irect[2], doc[0].get_pixmap().irect[3])

        c1, c2, c3 = st.columns(3)        
        
        with c1:
            pagenum = st.selectbox('select page', (range(1, len(doc)+1)))

        PDFParser().view(doc, dimension, pagenum)


if choose == '이미지':
    st.markdown(
        """
        <h3 style='text-align: center;'><br>이미지 OCR Parser</h3>
            
        <p style='text-align: center;'>이미지에서 텍스트, 작은 이미지, 표, 각주 및 레이아웃을 추출합니다.<p>
        """,
        unsafe_allow_html=True                
    )
    
    # file = st.file_uploader('upload image', type=['jpg', 'png', 'jpeg'])

    # if file is not None:

    #     IMAGE().view()

    

if choose == 'HTML 변환':
    st.markdown(
        """
        <h3 style='text-align: center;'><br>HTML 컨버터</h3>
            
        <p style='text-align: center;'>입력된 파일을 형태를 유지한 HTML로 변환합니다.<p>
        """,
        unsafe_allow_html=True                
    )
