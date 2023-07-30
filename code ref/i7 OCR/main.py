import streamlit as st
from stutils import load_css, add_logo, render_text
from streamlit_option_menu import option_menu
import streamlit_javascript as st_js
from parse import Parser
import fitz

# configure page
st.set_page_config(
    page_title="CrossConv",
    page_icon="arrows_clockwise",
    layout="wide",
)

load_css()
add_logo('l7_header.png')

# initialize session_state
if "ocr_request" not in st.session_state:
    st.session_state["ocr_request"] = False

if "pagenum" not in st.session_state:    
    st.session_state['pagenum'] = 1

if "file" not in st.session_state:    
    st.session_state['file'] = None

if 'downloaded' not in st.session_state:
    st.session_state['downloaded'] = False  

# set up layout & appearance
choose = option_menu("Document Cross Parser", ["PDF & Image"],
                        icons=['card-text'],
                        menu_icon="pencil-square", default_index=0,
                        orientation='horizontal',
                        styles={
    # "container": {"padding": "5!important", "background-color": "#fafafa"},
    "icon": {"color": "black", "font-size": "25px"}, 
    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "gray"},
    }
)

# render page
if choose == 'PDF & Image':    

    # get ui dimension in case adjustment needed
    if 'ui_width' not in st.session_state or \
        'device_type' not in st.session_state or \
            'device_width' not in st.session_state:
        
        ui_width = st_js.st_javascript("window.innerWidth", key="ui_width_comp")
        device_width = st_js.st_javascript("window.screen.width", key="device_width_comp")

        if ui_width > 0 and device_width > 0:

            if device_width > 768:
                device_type = 'desktop'
            else:
                device_type = 'mobile'

            st.session_state['ui_width'] = ui_width
            st.session_state['device_type'] = device_type
            st.session_state['device_width'] = device_width

            st.experimental_rerun()
    
    # render page
    else:

        st.markdown(
            """
            <p style='text-align: center;'>PDF 또는 이미지 파일에서 텍스트를 추출합니다.<p>
            """,
            unsafe_allow_html=True                
        )

        # upload file
        file = st.file_uploader('upload', type=['pdf', 'jpeg', 'png', 'jpg', 'gif'])

        # initialize session state if new file is uploaded
        if file != st.session_state.file:
            st.session_state['downloaded'] = False
            st.session_state['html_string'] = None
            st.session_state['full_text'] = None
            st.session_state['pagenum'] = 1
            st.session_state['dimension'] = None
            st.session_state["ocr_request"] = False
        st.session_state.file = file        

        # render file if uploaded
        if st.session_state.file is not None:

            filetype = file.name.split('.')[-1]

            # read file
            doc = fitz.open(stream=file.read(), filetype=filetype)
            dimension = (doc[0].get_pixmap().irect[2], doc[0].get_pixmap().irect[3])
            st.session_state['dimension'] = dimension

            # page selection dropdown
            c1, c2, c3 = st.columns(3)        
            
            with c1:                
                pagenum = st.selectbox(
                    'select page', 
                    (range(1, len(doc)+1))                    
                    )       
            
            # render view in 2 columns
            c11, c12 = st.columns(2)

            # if OCR requested                
            if st.session_state["ocr_request"] == True:  

                # column 1: render html w/ bbox
                with c11:
                    Parser().view(doc, pagenum)   

                # column 2: render text
                with c12:
                    render_text(st.session_state)                      

            # for new OCR request
            if st.session_state["ocr_request"] == False:  

                # column 1 only: render image + OCR reequest button
                with c11:
                    image = doc[pagenum-1].get_pixmap().tobytes('png')
                    st.image(image, width=dimension[0])

                    if st.button('OCR 요청'):            
                        st.session_state["ocr_request"] = True
                        st.experimental_rerun()                  
