# %%
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdftypes import resolve1
import warnings

def process_tag(tag, struct_elem):
    tag_type = resolve1(struct_elem.get('S')).name        
    # actual_text = resolve1(struct_elem.get('ActualText'))
    # print(f"Tag Type: {tag_type}, Actual Text: {actual_text}")
    print(f"Tag Type: {tag_type}, Structure Element: {struct_elem}")

def parse_tagged_pdf(file_path):
    with open(file_path, 'rb') as fp:
        # Create a PDF parser object associated with the file object
        parser = PDFParser(fp)
        # Create a PDF document object that stores the document structure
        doc = PDFDocument(parser)
        
        if not doc.info:
            warnings.warn("No metadata in the PDF")
        
        elif 'StructTreeRoot' not in doc.catalog:
            warnings.warn("The PDF is not tagged")

        else:
            struct_tree_root = resolve1(doc.catalog['StructTreeRoot'])
            # parent_struct_elem_list = resolve1(struct_tree_root['K'])
            struct_elems = resolve1(struct_tree_root['K'])

            for key in struct_elems.keys():
                tag_type = resolve1(struct_elems[key]).name
                if tag_type in ['H', 'P']:
                    process_tag(tag_type, struct_elems[key])                

            # for struct_elem in struct_elem_list:
            #     if isinstance(struct_elem, dict) and 'S' in struct_elem:
            #         tag_type = resolve1(struct_elem['S']).name
            #         if tag_type in ['H', 'P']:
            #             process_tag(tag_type, struct_elem)
            

pdf_path = "/home/lstm/Github/pdf2html/DATA_sample/보험/adobe/sample_insurance_policy.pdf"
parse_tagged_pdf(pdf_path)

# %%
