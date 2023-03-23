import os
import sys

import pdf_functionalities as pdf_funs
from flask import Flask, render_template, request, session

#
if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder,static_folder=static_folder)
else:
    app = Flask(__name__)


UPLOAD_FOLDER = os.path.join('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
upload_path = app.config['UPLOAD_FOLDER']
app.secret_key = 'This is your secret key to utilize session in Flask'



@app.route('/')
def html_home():
    home_page = pdf_funs.home()
    return home_page

@app.route('/upload_pdf', methods = ['POST','GET'])
def upload_pdf():
    result = pdf_funs.upload_pdf(request,path=upload_path)
    return result

@app.route('/getModel', methods = ['POST','GET'])
def choose_model():
    model_name = pdf_funs.getModel(request)
    return model_name

@app.route('/read')
def extract_text():
    pdf_text = pdf_funs.read_pdf()
    return pdf_text

@app.route('/fetch_details',methods = ['POST','GET'])
def fetch_data():
    result = pdf_funs.fetch_details()
    return result

@app.route('/add',methods=['POST','GET'])
def add_pairs():
    result = pdf_funs.add(request)
    return result

@app.route('/train_data',methods=['POST','GET'])
def update_pairs():
    result = pdf_funs.train_data(request)
    return result

@app.route('/train_model')
def export_pairs():
    result = pdf_funs.train_model()
    return result

@app.route('/export_tables')
def export_tables():
    result = pdf_funs.tables_ex()
    return result

@app.route("/choose_option",methods=['GET','POST'])
def choose_option():
    result = pdf_funs.choose_option(request)
    return result

@app.route('/create_model',methods=['GET','POST'])
def create_model():
    result = pdf_funs.create_model(request)
    return result

@app.route('/resume_training',methods=['GET','POST'])
def resume_training():
    result = pdf_funs.model_training(request)
    return result

if __name__ == '__main__':
    # app.run(host="127.0.0.2",debug=True)
    app.run(debug=True)