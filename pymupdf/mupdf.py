# %%
import fitz
fitz.__doc__
# %%
filename = '/home/lstm/Downloads/누락판례/image/done/특허법원_2017허2727_판결서'
# filename = '/home/lstm/Downloads/누락판례/image/done/특허법원_2016나1691_판결서'
# filename = '/home/lstm/Downloads/누락판례/text/대법원_2020도1007_판결서'
doc = fitz.open(filename + '.pdf')
doc
# %%
page4 = doc[3]
words = page4.get_text('words')
words
# %%
page4.annots()

for annot in page4.annots():

    if annot!=None:

        print(annot.rect)   

        # rec=annot.rect

        # mywords = [w for w in words if fitz.Rect(w[:4]) in rec]
# %%
page4.clean_contents()
for img in page1.get_images():
    print(img)

# %%

first_annots=[]
rec=page1.first_annot.rect
rec

# %%
#Information of words in first object is stored in mywords

mywords = [w for w in words if fitz.Rect(w[:4]) in rec]

ann= make_text(mywords)

first_annots.append(ann)

# %%
for pageno in range(0,len(doc)-1):

    page = doc[pageno]

    words = page.get_text("words")

    for annot in page.annots():

        if annot!=None:

            rec=annot.rect

            mywords = [w for w in words if fitz.Rect(w[:4]) in rec]

# %%
import io 
from PIL import Image

image_list = page4.get_images()
for idx, img in enumerate(image_list):
    xref = img[0]

    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]
    # get the image extension
    image_ext = base_image["ext"]
    # load it to PIL
    image = Image.open(io.BytesIO(image_bytes))
    # save it to local disk
    image.save(open(f"image{idx+1}_{idx}.{image_ext}", "wb"))

# %%
page4 = doc[3]
image_list = page4.get_images()
for idx, img in enumerate(image_list):
    xref = img[0]

    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]
    # get the image extension
    image_ext = base_image["ext"]
    # load it to PIL
    image = Image.open(io.BytesIO(image_bytes))
    # save it to local disk
    image.save(open(f"image7{idx+1}_{idx}.{image_ext}", "wb"))
# %%
print(page6.get_text())
# %%
page6.get_pixmap()
# %%
