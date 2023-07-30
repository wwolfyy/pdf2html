import streamlit as st
from stutils import detect_text_i7, produce_html, render_html
import base64
from dotenv import load_dotenv
import os


class Parser:

    # pagetitle = 'PDF & Image Parser'
    
    def view(_self, _doc, pagenum): 

        if st.session_state['downloaded']:

            render_html(st.session_state) 

        else:      

            load_dotenv()
            # url = "http://220.118.0.29:3516/predict_fastapi"
            url = os.getenv('OCR_URL')
            bytesio = _doc[pagenum-1].get_pixmap().tobytes('png') 

            resp = detect_text_i7(url, bytesio)    
            st.session_state['full_text'] = resp['full_text']              

            html_divs = ''
            for i, block in enumerate(resp['blocks']):

                if block['WordText'] != None:

                    html_divs = html_divs + (f"""
                    <div id="rect_{i}" style="position: absolute; 
                    left: {block['Left']}px; top: {block['Top']}px; 
                    width: {block['Width']}px; 
                    height: {block['Height']}px; 
                    border: 1px solid red; 
                    background-color: rgba(255, 0, 0, 0.3);"
                    title="{block['WordText']}"></div>
                    ></div>
                    """)
                                   
            b64image = base64.b64encode(bytesio).decode('utf-8')

            html_divs = f"""
            <div><img src="data:image/png;base64,{b64image}" style="position: absolute; 
            left: 0px; top: 0px;" 
            alt="image"></div>""" + html_divs

            # produce html code & render
            html_string = produce_html(html_divs)
            st.session_state['html_string'] = html_string
            render_html(st.session_state)   


