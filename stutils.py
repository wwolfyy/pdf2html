import streamlit as st
from PIL import Image
import time

def load_css():
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
        unsafe_allow_html=True)
    

def create_html_from_byteio(byteio):
    """Create HTML from a byteio object."""
    return f'<img src="data:image/png;base64,{byteio.getvalue()}" alt="image" width="100%" height="100%">'


def detect_text_gcv(bytesio):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    # with io.open(path, 'rb') as image_file:
    #     content = image_file.read()

    image = vision.types.Image(content=bytesio)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    return texts


def produce_html(htmlpath, imagepath, divs):

    html_string = f"""
        <html>
        <head>
        </head>
        <body>
        {divs}
        </body>
        </html>
        """
    with open(htmlpath, 'w') as f:
         f.write(html_string)

    return html_string


    