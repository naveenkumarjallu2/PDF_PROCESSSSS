import json


# Import the Images module from pillow
from PIL import Image
import PyPDF2
import spacy
import xlsxwriter
import pdfplumber
import pandas as pd
from flask import render_template, session

from werkzeug.utils import secure_filename

import os
# for spacy model training
import spacy
from pathlib import Path
import random
from spacy.training import Example
from spacy.util import minibatch
from tqdm import tqdm

dir = r"D:\project_pdf_profile\compelete_pdf_editor_model\pre_models"
models = []
for i in os.listdir(dir):
    models.append(i)

UPLOAD_FOLDER = os.path.join('static')
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
content = 'Read pdf Click On Read Button'
dict = {} # for text
d = {} # for tables
listOfPairs= []
selected_mdl = 'select_model'
training_data = []
added_pages = []
t_data = []

def home():
    return render_template('home_page.html')
def pdf_home():
    return render_template('index.html',content=content,len=len(listOfPairs),data=listOfPairs,models=models,selected_mdl='select model')

# for selecting pdf file from device
def upload_pdf(request,path):
    if request.method == 'POST':
        # f = request.files['file']
        # f.save(app.config['static'],f.filename)
        uploaded_pdf = request.files['file']
        # Extracting uploaded data file name
        pdf_filename = secure_filename(uploaded_pdf.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_pdf.save(os.path.join(path, pdf_filename))
        # Storing uploaded file path in custom_model session
        session['uploaded_pdf_file_path'] = os.path.join(path, pdf_filename)
        # print(session['uploaded_pdf_file_path'])
        # globals()['d'].clear()
        # globals()['dict'].clear()
        globals()['content'] = ''

        return render_template('index.html',content=content,len=len(listOfPairs),data=listOfPairs,user_pdf=session['uploaded_pdf_file_path'],models=models,selected_mdl=selected_mdl)

# for selecting model
def getModel(request):
    pdf_file_path = session.get('uploaded_pdf_file_path', None)
    if request.method == 'POST':
        model_name = request.form['ddlmodel']
        global selected_model
        selected_model = model_name
        global selected_mdl
        selected_mdl = model_name

    return render_template('index.html', content=content, len=len(listOfPairs), data=listOfPairs,
                           user_pdf=pdf_file_path,models=models,selected_mdl=selected_mdl)

# for extracting text from pdf
def read_pdf():
    pdf_file_path = session.get('uploaded_pdf_file_path', None)
    try:
        pdf = pdfplumber.open(pdf_file_path)
        l = pdf.pages
        text = ''
        for page in l:
            text += page.extract_text()+'\n'
        globals()['content'] = text
    except:
        global read_exception
        read_exception = 'Please Upload Pdf'
        return render_template('index.html', content=content, len=len(listOfPairs), data=listOfPairs,
                               user_pdf=pdf_file_path, models=models, selected_mdl=selected_mdl,model_exception=read_exception)



    return render_template('index.html',content = text,len=len(listOfPairs),data=listOfPairs,user_pdf=pdf_file_path,models=models,selected_mdl=selected_mdl)

def fetch_details():
    pdf_file_path = session.get('uploaded_pdf_file_path', None)
    # print('this is fetch fun',selected_model)
    try:
        print(selected_mdl)
        nlp_ner = spacy.load(rf'D:\project_pdf_profile\compelete_pdf_editor_model\pre_models\{selected_model}')
        # print('selected model path',rf'D:\python_projects\pdf_fun\models\{selected_model}'+'\n')
        doc = nlp_ner(content)
        # print('content is  : \n',content)
        temp = ''
        # print(doc.ents)
        for ent in doc.ents:
            l = []
            temp+=ent.label_+"  :  "+"\n"+ent.text+"\n\n"
            # print(temp)
        globals()['content']=temp
    except :
        global model_exception
        model_exception = 'Please Choose Model'
        return render_template('index.html', content=content, len=len(listOfPairs), data=listOfPairs,
                               user_pdf=session['uploaded_pdf_file_path'], models=models, selected_mdl=selected_mdl,model_exception=model_exception)

    return render_template('fetch.html', content=content, len=len(listOfPairs), data=listOfPairs, user_pdf=pdf_file_path,models=models,selected_mdl=selected_mdl)

# for adding lable and value to set

def add(request):
    pdf_file_path = session.get('uploaded_pdf_file_path', None)
    if request.method == 'POST':
        k = request.form['key']
        v = request.form['value']
        if k not in globals()['dict']:
            globals()['dict'][k]=[v]
        else:
            globals()['dict'][k] = globals()['dict'][k]+[v]
        list = []
        for i,j in globals()['dict'].items():
            if len(j) <= 1:
                list.append((i,j[0]))
            else:
                for vs in j:
                    list.append((i,vs))
        globals()['listOfPairs'] = list

# for storing lable and value with index pos we are taking Trained_data

    pdf_file_path = session.get('uploaded_pdf_file_path', None)
    pdf = pdfplumber.open(pdf_file_path)
    l = pdf.pages

    k = request.form['key']
    v = request.form['value'].replace('\r', '')
    value = v

    # print(value)
    for page_no in range(len(l)):
        page_text = l[page_no].extract_text()
        if value in page_text:
            start = page_text.find(value)
            length = len(value)
            end = start + length
            # print((start,end))
            # print(page_no)
            if page_no in added_pages:
                # print('this is added in to same page')
                for i in range(len(globals()['training_data'])):
                    if globals()['training_data'][i][0] == page_no:
                        globals()['training_data'][i][2].append((start, end, k))
                        break
                break
            else:
                # print('this is else')
                globals()['training_data'].append((page_no, l[page_no].extract_text(), [(start, end, k)]))
                globals()["added_pages"].append(page_no)
                break
    return render_template('index.html',content=content,len=len(listOfPairs),data=listOfPairs,user_pdf=pdf_file_path,models=models,selected_mdl=selected_mdl)
# for updating all the lable and values

def update(request):
    pdf_file_path = session.get('uploaded_pdf_file_path', None)
    if request.method == 'POST':
        temp={}
        for i in range(len(dict)):
            k = 'key'+str(i)
            v = 'value'+str(i)
            key = request.form[k]
            value = request.form[v]
            temp[key]=[value]
        globals()['dict'].clear()
        globals()['dict']=temp
        # print(globals()['dict'])
        list = []
        for i, j in globals()['dict'].items():
            list.append((i, j[0]))
        globals()['listOfPairs'] = list
    return render_template('index.html',content=content,len=len(listOfPairs),data=listOfPairs,user_pdf=pdf_file_path,models=models,selected_mdl=selected_mdl)


def export():
    pdf_file_path = session.get('uploaded_pdf_file_path', None)

    x = {}
    x.update(globals()['d'])
    x.update(globals()['dict'])
    # print(x)
    def pad_dict_list(dict_list, empty):
        lmax = 0
        for lname in dict_list.keys():
            lmax = max(lmax, len(dict_list[lname]))
        for lname in dict_list.keys():
            ll = len(dict_list[lname])
            if ll < lmax:
                dict_list[lname] += [empty] * (lmax - ll)
        return dict_list

    x = pad_dict_list(x, '')
    df = pd.DataFrame(x)
    # print(df)
    df.to_excel('static/training_data.xlsx',index=False)

    for i in globals()['training_data']:
        text = i[1]
        l_tuples = i[2]
        globals()['t_data'].append((text, {'entities': l_tuples}))

    data = json.dumps(globals()['t_data'])
    jsonFile = open(r"static/trained_data_new.json", "w")
    jsonFile.write(data)
    jsonFile.close()
    print(t_data)
    print(training_data)
    globals()['dict'].clear()

    return render_template('costum_model.html',tfn = False ,tfe = False,models = models,sld_md='select model')

# for extracting all the tables from pdf_file
def tables_ex():
    pdf_file_path = session.get('uploaded_pdf_file_path', None)

    pdf = pdfplumber.open(pdf_file_path)

    pages = pdf.pages
    workbook = xlsxwriter.Workbook('static/tender_tbls.xlsx')
    worksheet = workbook.add_worksheet()
    r_count = 0
    for i in range(len(pages)):
        # print(i)
        t_no = 1
        list = pdf.pages[i].extract_tables()
        for l in list:
            for row in range(len(l)):
                for col in range(len(l[row])):
                    worksheet.write(r_count, col, l[row][col])
                    # print([row], [col])
                r_count += 1
            r_count += 1
            worksheet.write(r_count, 0, "Table No" + str(t_no))
            t_no += 1

        # print(r_count)

    workbook.close()
    return render_template('index.html', content=content, len=len(listOfPairs), data=listOfPairs,
                           user_pdf=session['uploaded_pdf_file_path'],models=models,selected_mdl=selected_mdl)

# def train_model_home():
#     return render_template('costum_model.html',tfn = False ,tfe = False,models = ['ravi','teja'],sld_md='select model')


def choose_option(request):
    res = request.form.get('choose')
    if res == 'cm':
        return render_template('costum_model.html',tfn = True ,tfe = False,models = models,sld_md='select model')
    elif res == 'chm':
        return render_template('costum_model.html',tfn = False ,tfe = True,models = models,sld_md='select model')
    return render_template('costum_model.html', tfn=False, tfe=False, models=models, sld_md='select model',msg='please choose one option')


def model_name(request):
    global m_name
    m_name = request.form.get('model_name')
    # print(m_name)
    return render_template('costum_model.html', tfn=True, tfe=False, models=models, sld_md='select model')




def create_model(request):


    TRAIN_DATA = globals()['t_data']
    print('t_data create model:',t_data)
    nlp = spacy.blank('en')
    # output_dir = Path(rf"D:\project_pdf_profile\compelete_pdf_editor_model\models\{request.form.get('model_name')}")
    ner = nlp.create_pipe('ner')
    nlp.add_pipe('ner', last=True)
    ner = nlp.get_pipe('ner')
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        n_iter = 100
        optimizer = nlp.begin_training()

        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for batch in minibatch(TRAIN_DATA, size=8):
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    model_name = request.form.get('model_name')
    print(model_name)
    output_dir = rf'D:\project_pdf_profile\compelete_pdf_editor_model\pre_models\{model_name}'
    # Save the model to a directory using the nlp.to_disk function :
    nlp.to_disk(output_dir)
    print('model created')
    # add create model in to models list



    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print('Saved model to', output_dir)
        globals()['t_data'].clear()

    # add create model in to models list
    print('new model created')
    globals()['models'] = []
    for i in os.listdir(dir):
        globals()['models'].append(i)
    # nlp = spacy.blank('en')
    # ner = nlp.create_pipe('ner')
    # nlp.add_pipe('ner')
    # n_iter = 100
    # ner = nlp.get_pipe('ner')
    # print(ner)
    # # ner.add_label('ORG')
    # optimizer = nlp.begin_training()
    # for i in tqdm(range(n_iter)):
    #     losses = {}
    #     # ('text 3', {'cats': {'label 1': 0, 'label 2': 1, 'label 3': 0}})
    #     for text, annotations in TRAIN_DATA:
    #         # nlp.update([text], [annotations], sgd=optimizer, drop=0.2, losses=losses)
    #         doc = nlp.make_doc(text)
    #         example = Example.from_dict(doc, annotations)
    #         nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
    #         # print('Iteration', i, 'Loss', losses)
    #
    # # Save the model to a directory using the spacy.save_to_directory function
    # nlp.to_disk(output_dir)

    # globals()['models'] = []
    # for i in os.listdir(dir):
    #     globals()['models'].append(i)




    return render_template('costum_model.html', tfn=False, tfe=False, models=models, sld_md='select model',
                           msg='New Model is created successfully')



def model_training(request):

    TRAIN_DATA = globals()['t_data']
    print('re training data is ',TRAIN_DATA)

    # output_dir = Path(rf"D:\project_pdf_profile\compelete_pdf_editor_model\models\{request.form.get('ddlmodel')}")
    selected_model=request.form.get('ddlmodel')
    import spacy
#     nlp = spacy.load(rf'D:\project_pdf_profile\compelete_pdf_editor_model\pre_models\{selected_model}')
#     # Add new labels if necessary
#     ner = nlp.get_pipe('ner')
#     print('ner model training',ner)
#     n_iter =9
#     optimizer = nlp.resume_training()
#     for i in tqdm(range(n_iter)):
#         losses = {}
#         for text, annotations in TRAIN_DATA:
#             doc = nlp.make_doc(text)
#             example = Example.from_dict(doc, annotations)
#             nlp.update([example], drop=0.3, sgd=optimizer, losses=losses)
#             # print('Iteration', i, 'Loss', losses)
#     for text, _ in TRAIN_DATA:
#         doc = nlp(text)
#         print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
#
#
# # Save the model to a directory using the spacy.save_to_directory function:
#
#     output_dir = rf'D:\project_pdf_profile\compelete_pdf_editor_model\pre_models\{selected_model}'
#     nlp.to_disk(output_dir)

    nlp = spacy.load(rf'D:\project_pdf_profile\compelete_pdf_editor_model\pre_models\{selected_model}')

    ner = nlp.get_pipe('ner')

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        n_iter = 20
        optimizer = nlp.resume_training()

        for itn in tqdm(range(n_iter)):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for batch in minibatch(TRAIN_DATA, size=8):
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.25, sgd=optimizer, losses=losses)
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    output_dir = rf'D:\project_pdf_profile\compelete_pdf_editor_model\pre_models\{selected_model}'    # Save the model to a directory using the nlp.to_disk function :
    nlp.to_disk(output_dir)
    print('Model Retrained Successfully')
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print('Saved model to', output_dir)
        globals()['t_data'].clear()
        print(' model updated')


    return render_template('costum_model.html', tfn=False, tfe=False, models=models, sld_md='select model',
                           msg=' Model is updated successfully')



def extract_text(request):

    pdf_File = open(session['uploaded_pdf_file_path'], 'rb')

    # Create PDF Reader Object
    # pdf_Reader1 = PyPDF2.PdfReader(pdf_File)
    pdf_Reader = pdfplumber.open(pdf_File)
    # count = pdf_Reader.numPages # counts number of pages in pdf

    TextList = []
    def save_fun():
        print('this is save fun')
        if request.form.get('pagerange') == 'all':
            start = 0
            end = len(pdf_Reader.pages)
        else:
            start = request.form.get('start')
            end = request.form.get('end')
        return (start,end)
    if request.form.get('typeoftext') == 'text':
        start,end= save_fun()
        print('start',start,'end',end)
        for i in range(int(start) - 1, int(end)):

            page = pdf_Reader.pages[i]
            # print(page)

            TextList.append(page.extract_text())
            # Converting multiline text to single line text
            TextString = f"\n\n********************************************************************<page>********************************************************************\n\n".join(TextList)
            print(TextString)
            f = open("pdffile4.txt", 'w', encoding='utf-8')
            f.write(TextString)
            f.close()
    elif request.form.get('typeoftext') == 'table':
        start, end = save_fun()
        print('start', start, 'end', end)

        pages = pdf_Reader.pages
        workbook = xlsxwriter.Workbook('static/tender_tbls.xlsx')
        worksheet = workbook.add_worksheet()
        r_count = 0
        for i in range(int(start) - 1, int(end)):
            # print(i)
            t_no = 1
            list = pages[i].extract_tables()

            for l in list:

                r_count += 1
                t = worksheet.write(r_count, 0, f"\n\n\n ******.Page NO{i+1}..Table No" + str(t_no) +'...******\n')
                r_count += 1
                t_no += 1
                for row in range(len(l)):
                    for col in range(len(l[row])):
                        worksheet.write(r_count, col, l[row][col])
                    # print([row], [col])
                    r_count += 1


            # print(r_count)
        workbook.close()



    return 'done'


def load_pdf(request, path):
    uploaded_pdf = request.files['file']
    # Extracting uploaded data file name
    pdf_filename = secure_filename(uploaded_pdf.filename)
    # Upload file to database (defined uploaded folder in static path)
    uploaded_pdf.save(os.path.join(path, pdf_filename))
    session['uploaded_pdf_file_path'] = os.path.join(path, pdf_filename)
    return 'hi'


def test():

    return render_template('ex_text.html')

def testing():

    return render_template('ex_image.html')

def loading_pdf(request, path):
    uploaded_pdf = request.files['file']
    # Extracting uploaded data file name
    pdf_filename = secure_filename(uploaded_pdf.filename)
    # Upload file to database (defined uploaded folder in static path)
    uploaded_pdf.save(os.path.join(path, pdf_filename))
    session['uploaded_pdf_file_path'] = os.path.join(path, pdf_filename)
    return 'hi'

def extract_image(request):
    import fitz  # PyMuPDF
    import io
    from PIL import Image
    pdf_File = open(session['uploaded_pdf_file_path'], 'rb')
    # file path you want to extract images from

    # open the file
    pdf_file = fitz.open(pdf_File)

    # iterate over PDF pages
    if request.form.get('page') == 'all pages':
        start = 0
        end = len(pdf_file)
    else:
        start = request.form.get('start')
        end = request.form.get('end')
    for page_index in range(int(start) - 1, int(end)):
        # get the page itself
        page = pdf_file[page_index]
        # get image list
        image_list = page.get_images()
        # printing number of images found in this page
        if image_list:
            print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        else:
            print("[!] No images found on page", page_index)
        for image_index, img in enumerate(image_list, start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            # get the image extension
            # ext=request.form.get('ddlimagetype')
            # image_ext = base_image["ext"]
            # load it to PIL
            image = Image.open(io.BytesIO(image_bytes))
            # image.save(f"{request.form.get('Floder_path')}image_name{image_index}.{request.form.get('typeimg')}")
            image.save(f"{request.form.get('Floder_path')}/{request.form.get('imagename')}{image_index}{request.form.get('ddlimagetype')}",quality=int(request.form.get('quality')),xres=int(request.form.get('xres')),yres=int(request.form.get('yres')))
            # save it to local disk
            # image.save(open(f"images\image{page_index + 1}_{image_index}.{image_ext}", "wb"))



    return 'image'


def convert_model(request):
    model_path=request.form.get('i_path')
    for i in os.listdir(dir):
        globals()['models'].append(i)

    return 'model'


def type_model():
    return render_template('ex_model.html')
