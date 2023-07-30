import streamlit as st
import uuid
import base64
import re
import json
import requests    


def load_css():

    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
    st.markdown(
        """
        <link rel="stylesheet" 
        href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
        integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
        crossorigin="anonymous">
        """,
        unsafe_allow_html=True
        )


def detect_text_i7(url=None, bytesio=None):    
    """
    Send OCR request to l7 server
    input: 
        url: string
        bytesio: io.BytesIO object (image file)
    output: dictionary
        full_text: string
        blocks: rectangle coordinates & text
    """

    response = requests.post(url, files={"file": bytesio})

    status = response.status_code            
                
    if status == 200:
        print('success')
    else:
        msg = 'response code: ' + str(status)
        print(msg)
        return msg
    
    resp = response.json()
    full_text = resp['ParsedResults'][0]['ParsedText']

    raw_blocks = resp['ParsedResults'][0]['Overlay']['Lines']
    block_list = [block['Words'] for block in raw_blocks]

    # flatten word_blocks list
    blocks = []
    for block in block_list:
        for word in block:
            blocks.append(word)

    return {'full_text': full_text, 'blocks': blocks}


def produce_html(divs):

    html_string = f"""
        <html>
        <head>
        </head>
        <body style="background-color:white;">
        {divs}
        </body>
        </html>
        """

    return html_string

    
def render_html(session_state):

    st.components.v1.html(
            session_state['html_string'], 
            session_state['dimension'][0], 
            session_state['dimension'][1], 
            scrolling=False
            )


def render_text(session_state):    

    html_divs = f"""
    <div style="width:overflow:scroll; width:{session_state['dimension'][1]}; height:{session_state['dimension'][0]};"><pre>
    """ + session_state['full_text'] + '</pre></div>'
    # print(session_state['full_text'])

    st.session_state['html_text'] = produce_html(html_divs)

    st.components.v1.html(
            session_state['html_text'], 
            session_state['dimension'][0], 
            session_state['dimension'][1], 
            scrolling=True
            )   

    download_button_html = download_button(
        st.session_state['full_text'].encode('utf-8'), 
        'ocrtxt.txt', 
        'Download'
        )
    
    st.markdown(download_button_html, unsafe_allow_html=True)    


def download_button(object_to_download, download_filename, button_text):

    if isinstance(object_to_download, bytes):
        pass

    else:
        object_to_download = json.dumps(object_to_download)

    try:        
        b64 = base64.b64encode(object_to_download.encode('utf-8')).decode('utf-8')

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link      


@st.cache_data
def add_logo(filepath):
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    image64 = base64.b64encode(data).decode()
    
    page_bg_img = '''
    <style>
    [data-testid="stHeader"] {
    background-image: url("data:image/png;base64,%s");
    background-repeat: no-repeat;
    background-size: contain;
    background-position: center;
    padding-top: 140px;        
    }
    </style>
    ''' % image64
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

