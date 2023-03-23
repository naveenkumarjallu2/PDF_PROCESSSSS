#Importing PDF reader PyPDF2
import os

import PyPDF2
from flask import session
from werkzeug.utils import secure_filename

#Open file Path
pdf_File = open(r"D:\project_pdf_profile\compelete_pdf_editor_model\static\tws_tender_doc_05082022.pdf", 'rb')

#Create PDF Reader Object
pdf_Reader = PyPDF2.PdfReader(pdf_File)
# count = pdf_Reader.numPages # counts number of pages in pdf
srt=int(input('enter starting page '))
end=int(input('enter end page '))
TextList = []
#Extracting text data from each page of the pdf file

for i in range(srt-1,end):
    page = pdf_Reader.pages[i]
    TextList.append(page.extract_text())
#Converting multiline text to single line text
TextString = f"\n\n********************************************************************<page{i}>********************************************************************\n\n".join(TextList)
print(TextString)



# EXTRACT_TEXT



def extract_text(request,path,start,end):

    uploaded_pdf = request.files['file']
    # Extracting uploaded data file name
    pdf_filename = secure_filename(uploaded_pdf.filename)
    # Upload file to database (defined uploaded folder in static path)
    uploaded_pdf.save(os.path.join(path, pdf_filename))
    session['uploaded_pdf_file_path'] = os.path.join(path, pdf_filename)
    pdf_File = open(session['uploaded_pdf_file_path'], 'rb')

    # Create PDF Reader Object
    pdf_Reader = PyPDF2.PdfReader(pdf_File)
    # count = pdf_Reader.numPages # counts number of pages in pdf

    TextList = []
    # Extracting text data from each page of the pdf file

    for i in range(int(start) - 1, int(end)):
        page = pdf_Reader.pages[i]
        TextList.append(page.extract_text())
    # Converting multiline text to single line text
    TextString = f"\n\n********************************************************************<page{i}>********************************************************************\n\n".join(TextList)
    print(TextString)
    f = open(path,'w')
    f.write(TextString)
    f.close()
    return
