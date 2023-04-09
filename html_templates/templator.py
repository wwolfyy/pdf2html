from PIL import Image
import base64

# Load the image file
# img = Image.open("image1.jpg")

# convert img to base64 STRING
with open("image1.jpg", "rb") as image_file:
    img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# # Load the HTML template from a file or string
with open("template.html", "r") as f:
    template = f.read()

with open("templatestyle.html", "r") as f:
    style = f.read()

# Substitute the `img1` placeholder with the base64-encoded image data
html = template.format(img1=img_base64)
html = html.replace("<style></style>", style)

# Write the HTML code to a file
with open("output.html", "w") as f:
    f.write(html.strip())

