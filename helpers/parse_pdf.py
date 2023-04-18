import fitz

# read pdf
doc_path = '/home/lstm/Github/pdf2html/DATA_sample/보험/sample_insurance_policy.pdf'
doc = fitz.open(doc_path)

# show page 53 content
page = doc[52]
print(page.get_text())
