import json
import xlsxwriter
import pdfplumber
import pandas as pd
from flask import render_template, session
import json
from werkzeug.utils import secure_filename
# pyinstaller  --onefile --hidden-import xlsxwriter --hidden-import pdfplumber --hidden-import pandas --hidden-import spacy --hidden-import random --clean  controller.py
import os
# for spacy model training
import spacy
from pathlib import Path
import random
from spacy.training import Example
from spacy.util import minibatch
from tqdm import tqdm

dir = r"D:\project_pdf_profile\compelete_pdf_editor_model\pre_model"
models = []
for i in os.listdir(dir):
    models.append(i)

UPLOAD_FOLDER = os.path.join('static')
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

selected_mdl = 'select model'
listOfPairs = []
def pad_dict_list(dict_list, empty):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if ll < lmax:
            dict_list[lname] += [empty] * (lmax - ll)
    return dict_list

def home():
    # , data = listOfPairs,content=content

    return render_template('index.html',len=len(listOfPairs),models=models,selected_mdl='selected_model',data=listOfPairs)

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
        global current_pdf_path
        current_pdf_path = session['uploaded_pdf_file_path']
        # print(session['uploaded_pdf_file_path'])
        global content
        content = 'click read button for extract text'
        global tables
        tables = {}
        globals()['listOfPairs'] = []

        return render_template('index.html',len=len(listOfPairs),user_pdf=current_pdf_path,models=models,selected_mdl='selected_model')

# for selecting model
def getModel(request):

    if request.method == 'POST':
        model_name = request.form['ddlmodel']
        global selected_mdl
        selected_mdl = model_name

    return render_template('index.html', content=content,user_pdf=current_pdf_path,
                           len=len(listOfPairs),models=models,selected_mdl=selected_mdl,data=listOfPairs)

# for extracting text from pdf
def read_pdf():
    if current_pdf_path == '':
        return render_template('index.html',len=len(listOfPairs),models=models,
                               selected_mdl=selected_mdl,model_exception = 'please upload pdf')
    else:
        if content == 'Read pdf click read button':
            pdf = pdfplumber.open(current_pdf_path)
            l = pdf.pages
            text = ''
            for page in tqdm(l):
                text += page.extract_text()+'\n'
            globals()['content'] = text
            return render_template('index.html',content = content,len=len(listOfPairs),
                                   user_pdf=current_pdf_path,models=models,selected_mdl=selected_mdl)
        else:
            return render_template('index.html', content=content, len=len(listOfPairs),data=listOfPairs,
                                   user_pdf=current_pdf_path, models=models, selected_mdl=selected_mdl)

def fetch_details():
    if selected_mdl == 'select model':
        model_exception = 'Please Choose Model'
        return render_template('index.html',model_exception=model_exception,content=content, len=len(listOfPairs),
                               user_pdf=current_pdf_path, models=models,selected_mdl=selected_mdl)
    else:
        nlp_ner = spacy.load(rf'models\{selected_mdl}')
        doc = nlp_ner(content)
        temp = ''
        # print(doc.ents)
        for word in doc.ents:
            temp += word.label_ + "  :  " + "\n" + word.text + "\n\n"
    return render_template('fetch.html',content=temp,user_pdf=current_pdf_path,models=models,selected_mdl=selected_mdl)

def add(request):
    if request.method == 'POST':
        t = request.form.get('ex_text')
        k = request.form.get('key')
        v = request.form.get('value')
        globals()['listOfPairs'].append((t,k,v))

    return render_template('index.html',content=content,len=len(listOfPairs),
                           data=listOfPairs,user_pdf=current_pdf_path,
                           models=models,selected_mdl=selected_mdl)

def train_data(request):

    temp=[]
    temp_dict = {}
    for i in range(len(listOfPairs)):
        text = request.form.get('ex_text'+str(i))
        key = request.form.get('key'+str(i))
        value = request.form.get('value'+str(i))
        main = text
        sub = value
        start = main.find(sub)
        end = start+len(sub)
        if temp:
            for i in range(len(temp)):
                if temp[i][0] == text:
                    temp[i][1].get('entities').append((start,end,key))
                else:
                    temp.append((text, {'entities': [(start, end, key)]}))
        else:
            temp.append((text, {'entities': [(start, end, key)]}))

        # for storing train data into dictionary
        if key not in temp_dict:
            temp_dict[key] = [value]
        else:
            temp_dict[key] = temp_dict[key]+[value]

    temp_dict = pad_dict_list(temp_dict, '')

    df = pd.DataFrame(temp_dict)
    df.to_excel(r"static/train_data.xlsx")
    print('labels and values stored in excel')

    data = json.dumps(temp)
    jsonFile = open(r"static/train_data.json", "w")
    jsonFile.write(data)
    jsonFile.close()
    print('train data stored in format of json')
    globals()['listOfPairs'] = []
    # for storing into Excel file



    return render_template('index.html',content=content,len=len(listOfPairs),data=listOfPairs,
                           user_pdf=current_pdf_path,models=models,selected_mdl=selected_mdl)


def train_model():

    return render_template('costum_model.html',tfn = False ,tfe = False,models = models,sld_md='select model')

# for extracting all the tables from pdf_file
def tables_ex():
    pdf = pdfplumber.open(current_pdf_path)
    pages = pdf.pages
    workbook = xlsxwriter.Workbook('static/extracted_tables.xlsx')
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
    workbook.close()
    return render_template('index.html', content=content, len=len(listOfPairs), data=listOfPairs,
                           user_pdf=current_pdf_path,models=models,selected_mdl=selected_mdl)

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
    import json
    # Opening JSON file
    f = open(r'static\train_data.json')

    # returns JSON object as a dictionary
    train_data = json.load(f)
    # take a black model or creating a new model
    nlp = spacy.blank('en')
    # creating ner pipe
    ner = nlp.create_pipe('ner')
    # add ner pipe to model
    nlp.add_pipe('ner',last=True)
    # get ner pipe
    ner = nlp.get_pipe('ner')
    # for adding labels to ner
    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    # for disable unwanted pipes
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        #set number of iterations
        n_iter = 100
        # start training for first time
        optimizer = nlp.begin_training()
        # loop for iterations
        for i in tqdm(range(n_iter)):
            # shuffle train data for each iteration
            random.shuffle(train_data)
            losses = {}
            # creating a minibatch with train data
            for batch in minibatch(train_data, size=8):
                # for getting text and annotations from batch
                for text, annotations in batch:
                    # add text into nlp as a doc
                    doc = nlp.make_doc(text)
                    # adding annotations and text to examples
                    example = Example.from_dict(doc, annotations)
                    # update nlp this training data
                    nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)

    # for getting output directory from clint server
    model_name = request.form.get('model_name')
    print(model_name)
    output_dir = f'models/{model_name}'
    # Save the model to a directory using the nlp.to_disk function :
    nlp.to_disk(output_dir)
    print('model created')
    # add create model in to models list
    globals()['models'] = []
    for i in os.listdir(dir):
        globals()['models'].append(i)
    # Closing json file
    f.close()

    return render_template('costum_model.html', tfn=False, tfe=False, models=models, sld_md='select model',
                           msg='New Model is created successfully')


def model_training(request):
    # Opening JSON file
    f = open(r'static\train_data.json')

    # returns JSON object as a dictionary
    train_data = json.load(f)
    # getting selected model from clint sever
    selected_model = request.form.get('ddlmodel')
    # loading existing model for retaining
    nlp = spacy.load(fr'models/{selected_model}')
    # get ner pipe
    ner = nlp.get_pipe('ner')
    # for adding labels to ner
    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    # for disable unwanted pipes
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        # set iterations
        n_iter = 100
        # start retraining existing model with resume_training()
        optimizer = nlp.resume_training()
        # loop for iterations
        for i in tqdm(range(n_iter)):
            # shuffle train data for each iteration
            random.shuffle(train_data)
            losses = {}
            # creating a minibatch with train data
            for batch in minibatch(train_data, size=8):
                # for getting text and annotations from batch
                for text, annotations in batch:
                    # add text into nlp as a doc
                    doc = nlp.make_doc(text)
                    # adding annotations and text to examples
                    example = Example.from_dict(doc, annotations)
                    # update nlp this training data
                    nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)

    # set output directory for retrained model
    output_dir = f'models/{selected_model}'
    # Save the model to a directory using the nlp.to_disk function :
    nlp.to_disk(output_dir)
    print('Model Retrained Successfully')
    # Closing json file
    f.close()

    return render_template('costum_model.html', tfn=False, tfe=False, models=models, sld_md='select model',
                           msg=' Model is updated successfully')

