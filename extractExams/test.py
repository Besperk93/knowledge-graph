from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTFigure, LTTextBox
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser

text = ""
filename = "AQA-83001F-QP-NOV20.PDF"
pdf_path = "resources/" + filename

with open(pdf_path, 'rb') as infile:
    parser = PDFParser(infile)
    doc = PDFDocument(parser)
    pages = list(PDFPage.create_pages(doc))
    for page in pages:
        rsrcmgr = PDFResourceManager()
        device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        interpreter.process_page(page)
        layout = device.get_result()

        for obj in layout:
            if isinstance(obj, LTTextBox):
                text += obj.get_text()

            elif isinstance(obj, LTFigure):
                print("Skipping figure")

with open("output/" + filename, "w") as outfile:
    outfile.write(text)
