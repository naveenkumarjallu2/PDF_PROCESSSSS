import os
import pdf_functionalities1 as pdf_funs
from flask import Flask, render_template, request, session

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
upload_path = app.config['UPLOAD_FOLDER']
app.secret_key = 'This is your secret key to utilize session in Flask'

@app.route('/')
def home():
    home=pdf_funs.home()
    return home


@app.route('/home')
def html_home():
    home_page = pdf_funs.pdf_home()
    return home_page

@app.route('/success', methods = ['POST','GET'])
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

@app.route('/update',methods=['POST','GET'])
def update_pairs():
    result = pdf_funs.update(request)
    return result

@app.route('/export')
def export_pairs():
    result = pdf_funs.export()
    return result

@app.route('/export_tables')
def export_tables():
    result = pdf_funs.tables_ex()
    return result

# @app.route("/train_model")
# def train_model():
#     result= pdf_funs.train_model_home()
#     return result

@app.route("/choose_option",methods=['GET','POST'])
def choose_option():
    result = pdf_funs.choose_option(request)
    return result
#
# @app.route("/model_name",methods=['GET','POST'])
# def model_name():
#     result = pdf_funs.model_name(request)
#     return result
@app.route('/create_model',methods=['GET','POST'])
def create_model():
    result = pdf_funs.create_model(request)
    return result
@app.route('/resume_training',methods=['GET','POST'])
def resume_training():
    result = pdf_funs.model_training(request)
    return result

@app.route('/test')
def test():
    result = pdf_funs.test()
    return result

@app.route('/load_pdf',methods=['GET','POST'])
def load_pdf():

    result = pdf_funs.load_pdf(request,path=upload_path)
    return test()


@app.route('/test1',methods=['GET','POST'])
def test1():

    result = pdf_funs.extract_text(request)
    return test()

@app.route('/testing')
def testing():
    result = pdf_funs.testing()
    return result

@app.route('/loading_pdf',methods=['GET','POST'])
def loading_pdf():
    result = pdf_funs.loading_pdf(request,path=upload_path)
    return testing()

@app.route('/testing1',methods=['GET','POST'])
def testing1():
    result = pdf_funs.extract_image(request)
    return testing()

@app.route('/type_model')
def type_model():
    result = pdf_funs.type_model()
    return result


@app.route('/convert_model',methods=['GET','POST'])
def convert_model():
    result=pdf_funs.convert_model(request)
    return result
# @app.route('/type_model',methods=['GET','POST'])
# def type_model():
#     resul


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
    # app.run(debug=True)