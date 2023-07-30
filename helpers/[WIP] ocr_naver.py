import requests
import uuid
import time
import json
import os
import fitz
from io import BytesIO, BufferedReader
from PIL import Image

# doc path
folderpath = "../DATA/training_data/leg_trend/Issue_pdf/"
doclist = os.listdir(folderpath)
print(doclist)
docindex = 6
docpath = os.path.join(folderpath, doclist[docindex])
print(doclist[docindex])


def ocr_naver(docpath):

    api_url = 'https://v59hbzqfj9.apigw.ntruss.com/custom/v1/23381/3799650a902178351e1986c326adf0fdb4249c3e3282e6b8c07fc9412dee6e24/general'
    secret_key = 'SVZpb0p5WW1QSERvaGJ4ekl4elRDTkN1alJUTUV3b0c='
    doc = fitz.open(docpath)

    # page = doc.load_page(0)
    rotation = doc.load_page(0).rotation
    if not rotation == 0:
        raise ValueError('현재는 세로 방향의 문서만 지원합니다.')

    # image_file = docpath
    files = []
    for pagenum, page in enumerate(doc):
        page_pixmap = page.get_pixmap()
        page_image_bytes = page_pixmap.tobytes('png')
        page_image_buffered = BufferedReader(BytesIO(page_image_bytes))
        files.append(('file', page_image_buffered))

        if pagenum > 2:
            break

    request_json = {
        'images': [
            {
                'format': 'png',
                'name': doclist[docindex]
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    # files = [('file', open(image_file,'rb'))]
    headers = {
        'X-OCR-SECRET': secret_key
    }
    response = requests.request("POST", api_url, headers=headers, data=payload, files = files[2:3])

    # get response string
    resp_string = response.text

    # convert response string to dict
    resp_dict = json.loads(resp_string)


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

    html_divs = ''
    for i, block in enumerate(resp_dict['images'][0]['fields']):

        html_divs = html_divs + (f"""
        <div id="rect_{i}" style="position: absolute;
        left: {block['boundingPoly']['vertices'][0]['x']}px;
        top: {block['boundingPoly']['vertices'][0]['y']}px;
        width: {block['boundingPoly']['vertices'][1]['x'] - block['boundingPoly']['vertices'][0]['x']}px;
        height: {block['boundingPoly']['vertices'][2]['y'] - block['boundingPoly']['vertices'][1]['y']}px;
        border: 1px solid red;
        background-color: rgba(255, 0, 0, 0.3);"
        title="{block['WordText']}"></div>
        ></div>
        """)


    def produce_html(words):

        html_string = f"""
            <html>
            <head>
            </head>
            <body style="background-color:white;">
            """

        for word in words:
            html_string += f"""
                <span style="position: absolute;
                left: {word['x']}px;
                top: {word['y']}px;
                width: {word['width']}px;
                height: {word['height']}px;
                border: 1px solid red;
                background-color: rgba(255, 0, 0, 0.3);">
                {word['text']}
                </span>
            """

        html_string += """
            </body>
            </html>
            """

        return html_string








    b64image = base64.b64encode(bytesio).decode('utf-8')

    html_divs = f"""
    <div><img src="data:image/png;base64,{b64image}" style="position: absolute;
    left: 0px; top: 0px;"
    alt="image"></div>""" + html_divs

    # produce html code & render
    html_string = produce_html(html_divs)
