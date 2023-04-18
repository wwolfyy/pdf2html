from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal, LTChar, LTText, LTLine, LTRect, LTCurve, LTImage, LTAnno, LTTextGroup, LTTextBox, LTTextLine

doc = open
pages = extract_pages('/home/lstm/Github/pdf2html/DATA_sample/판례/adobe/특허법원 2017허8459.pdf')
dir(pages)

for page_layout in pages:
    for element in page_layout:
        print(element)



from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
output_string = StringIO()
with open('/home/lstm/Github/pdf2html/DATA_sample/판례/adobe/특허법원 2017허8459.pdf', 'rb') as fin:
    extract_text_to_fp(fin, output_string, laparams=LAParams(),
                       output_type='html', codec=None)


html = extract_text_to_fp('/home/lstm/Github/pdf2html/DATA_sample/판례/adobe/특허법원 2017허8459.pdf', output_string, laparams=LAParams(),
                       output_type='html', codec=None)


html = extract_text('/home/lstm/Github/pdf2html/DATA_sample/판례/adobe/특허법원 2017허8459.pdf')
print(html)





from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice

# Open a PDF file.
fp = open('/home/lstm/Github/pdf2html/DATA_sample/보험/sample_insurance_policy.pdf', 'rb')
# Create a PDF parser object associated with the file object.
parser = PDFParser(fp)
# Create a PDF document object that stores the document structure.
# Supply the password for initialization.
document = PDFDocument(parser)
# Check if the document allows text extraction. If not, abort.
if not document.is_extractable:
    raise PDFTextExtractionNotAllowed
# Create a PDF resource manager object that stores shared resources.
rsrcmgr = PDFResourceManager()
# Create a PDF device object.
device = PDFDevice(rsrcmgr)
# Create a PDF interpreter object.
interpreter = PDFPageInterpreter(rsrcmgr, device)
# Process each page contained in the document.
for page in PDFPage.create_pages(document):
    interpreter.process_page(page)





from pdfminer.layout import LAParams
from pdfminer.converter import PDFResourceManager, PDFPageAggregator
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LTTextBoxHorizontal

document = open('/home/lstm/Github/pdf2html/DATA_sample/보험/adobe/sample_insurance_policy.pdf', 'rb')
#Create resource manager
rsrcmgr = PDFResourceManager()
# Set parameters for analysis.
laparams = LAParams()
# Create a PDF page aggregator object.
device = PDFPageAggregator(rsrcmgr, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)

font_attributes = []
header_atrtributes = []
for page in PDFPage.get_pages(document):
    interpreter.process_page(page)
    # receive the LTPage object for the page.
    layout = device.get_result()
    for element in layout:
        if isinstance(element, LTTextBoxHorizontal):
            # get font name, size, color
            font_name = element.get_fontname()
            font_size = element.get_fontsize()

            
    break



# ======== get pdf outline ========
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

# Open a PDF document.
fp = open('/home/lstm/Github/pdf2html/DATA_sample/판례/adobe/특허법원 2017허8459.pdf', 'rb')
parser = PDFParser(fp)
document = PDFDocument(parser)

# Get the outlines of the document.
outlines = document.get_outlines()
for (level,title,dest,a,se) in outlines:
    print (level, title)



