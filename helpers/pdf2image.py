import fitz

# open pdf doc and convert to image. image path is in parent folder
# docpath = '/home/lstm/Github/pdf2html/부산지법_2018나55364_판결서.pdf'
docpath = '/home/lstm/Github/pdf2html/DATA_sample/보험/sample_insurance_policy.pdf'
doc = fitz.open(docpath)
# page 2 from doc
page = doc[45]
pix = page.get_pixmap()
pix.save('image_insurance.png')