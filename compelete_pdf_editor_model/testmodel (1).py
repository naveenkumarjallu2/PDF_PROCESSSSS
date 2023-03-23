from importlib.resources import path
from pathlib import Path
from tqdm import tqdm
import spacy
from spacy.training import Example
# catastrophic forgetting problem
#Add your custom pipeline components to the nlp object. For example, you can add a text classification component as follows:
# ner = nlp.create_pipe('ner')
# nlp.add_pipe('ner')
#Train the model using your training data:
# train_data = [('Walmart is a leading e-commerce company I reached Chennai yesterday. I recently ordered a book from Amazon', {'Walmart is a leading e-commerce company': {'label 1': 1, 'label 2': 0}}), ('text 2', {'cats': {'label 1': 0, 'label 2': 1}})]
model = 'D:\project_pdf_profile\compelete_pdf_editor_model\pre_model'
def train():
    nlp = spacy.blank('en')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe('ner')
    train_data = [("Flipkart is recognized as leader in market",{"entities": [(0,8, "ORG")]}),
        ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "COMPANY")]}),
                  ("Naveen is a leading IT company", {"entities": [(0, 6, "BOY")]}),
        ]
    n_iter = 100
    ner = nlp.get_pipe('ner')
    print(ner)
    # ner.add_label('ORG')
    optimizer = nlp.begin_training()
    for i in tqdm(range(n_iter)):
        losses = {}
        # ('text 3', {'cats': {'label 1': 0, 'label 2': 1, 'label 3': 0}})
        for text, annotations in train_data:
            # print('text',text)
            # print('annotaions',annotations)

            # nlp.update([text], [annotations], sgd=optimizer, drop=0.2, losses=losses)
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
            # print('Iteration', i, 'Loss', losses)

     #Save the model to a directory using the spacy.save_to_directory function:
    output_dir = model
    nlp.to_disk(output_dir)


def retrain():


    import spacy
    nlp = spacy.load(model)
    # Add new labels if necessary
    ner = nlp.get_pipe('ner')
    print(ner)
    # ner.add_label('FOOD')
    # Train the model
    train_data = [
                  # ("BMW is recognized as leader in market", {"entities": [(0, 3, "ORG")]}),
                  # ("Apple is recognized as leader in market", {"entities": [(0, 5, "ORG")]}),
            ("Bottel can be ordered in Amazon ", {"entities": [(0,6, "PRODUCT")]}),
        ("kumar is a building spacy model", {"entities": [(0, 5, "BOY")]}),
            ("Bags seals in store ", {"entities": [(0,4, "PRODUCT")]}),
                  ("gobi is a common fast food.", {"entities": [(0, 4, "RECIPE")]}),
        ("buymores is recognized as leader in market", {"entities": [(0, 8, "ORG")]}),
                  ("Pasta is an italian recipe", {"entities": [(0, 5, "FOOD")]}),
        ("Athamas ITsolution is a leading e-commerce", {"entities": [(0, 7, "SOFT")]}),
        ("raviteja is a good friend", {"entities": [(0, 8, "PERSON")]}),
                ("hari loves grills", {"entities": [(0,4,"MALE"),(11, 17, "FEMALE")]}),

                  ]
    train_data=[('3M Health Care Invoice\n100 3M Way\nSt. Paul\n55144\nUnited States\nBill To H1- Retail Invoice Number 2001528\n2220 Bridgepointe Pkwy Date 2/8/2022\nSan Mateo, CA 94404 PO Number 7111\nUnited States\nDescription Quantity Unit price Amount\nPLATE BOTTOM 0001 3582 6 $1,750.00 $10,500.00\nSHACKLE 0002 LONG 2984 5 $2,876.00 $14,380.00\nLOCK BODY 0001 3502 4 $1,567.00 $6,268.00\nTotal $31,148.00', {'entities': [(97, 104, 'Invoice number '), (23, 33, '3M')]})]
    # train_data = [("naveen is a common Good Guy", {"entities": [(0, 6, "PERSON")]}),
    #           ("raviteja is a good friend", {"entities": [(0, 8, "PERSON")]}),
    #               ("Athamas ITsolution is a leading e-commerce", {"entities": [(8, 14, "soft")]}),]
    n_iter =5
    optimizer = nlp.resume_training()
    for i in tqdm(range(n_iter)):
        losses = {}
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=-1, sgd=optimizer, losses=losses)
            # print('Iteration', i, 'Loss', losses)


            # Save the model to a directory using the spacy.save_to_directory function:
    output_dir = 'D:\project_pdf_profile\compelete_pdf_editor_model\models\model'
    output_dir = Path(output_dir)
    nlp.to_disk(output_dir)



def test():
    nlp = spacy.load(r'D:\project_pdf_profile\compelete_pdf_editor_model\models\model')
    doc = nlp('3M Health Care Invoice\n100 3M Way\nSt. Paul\n55144\nUnited States\nBill To H1- Retail Invoice Number 2001528\n2220 Bridgepointe Pkwy Date 2/8/2022\nSan Mateo, CA 94404 PO Number 7111\nUnited States\nDescription Quantity Unit price Amount\nPLATE BOTTOM 0001 3582 6 $1,750.00 $10,500.00\nSHACKLE 0002 LONG 2984 5 $2,876.00 $14,380.00\nLOCK BODY 0001 3502 4 $1,567.00 $6,268.00\nTotal $31,148.00')
    # print(doc.ents)
    for words in doc.ents:
        print(words.text,words.label_)
#
# train()
# #
retrain()
test()
# ("I recently ordered a book from Amazon", {"entities": [(31,37, "ORG")]})
# ("I rented a camera", {"entities": [(11,17, "PRODUCT")]}),
#  ("Fridge can be ordered in Amazon ", {"entities": [(0,6, "PRODUCT")]})
s = "3M Health Care Invoice\n100 3M Way\nSt. Paul\n55144\nUnited States\nBill To H1- Retail Invoice Number 2001528\n2220 Bridgepointe Pkwy Date 2/8/2022\nSan Mateo, CA 94404 PO Number 7111\nUnited States\nDescription Quantity Unit price Amount\nPLATE BOTTOM 0001 3582 6 $1,750.00 $10,500.00\nSHACKLE 0002 LONG 2984 5 $2,876.00 $14,380.00\nLOCK BODY 0001 3502 4 $1,567.00 $6,268.00\nTotal $31,148.00"
# print(s.find('100 3M Way'))
# print(len('100 3M Way'))
# import spacy
# nlp = spacy.load(model)
# # Add new labels if necessary
# ner = nlp.get_pipe('ner')
# print(ner)
# import spacy
#
# nlp = spacy.load(model)
# doc = nlp(u"Pasta is an italian recipe")

# revision_data = []
# ner = nlp.get_pipe('ner')
# print(ner)
# for d in nlp.pipe('ner'):
#     print(d)
# Apply the initial model to raw examples. You'll want to experiment
# with finding a good number of revision texts. It can also help to
# filter out some data.
# for doc in nlp.pipe([("Pasta is an italian recipe"),(u"Pasta is an italian recipe")]):
#     tags = [w.tag_ for w in doc]
#     heads = [w.head.i for w in doc]
#     deps = [w.dep_ for w in doc]
#     labs = [w.textaa for w in doc]
#     print(labs)
#     print(tags)
#     print(heads)
#     print(deps)
    # entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    # revision_data.append((doc, GoldParse(doc, tags=doc_tags, heads=heads,
    #                                         deps=deps, entities=entities)))
