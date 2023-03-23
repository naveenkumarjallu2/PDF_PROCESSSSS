from tqdm import tqdm


# _____________________________________________________________________________________________________________________
# _______________________________________________________
import random
import spacy
# nlp = spacy.blank('en')  # create blank Language class

# start_training = spacy.load("spacy_start_model")

# TRAIN_DATA = [
#               ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]}),
#               ("I reached Chennai yesterday.", {"entities": [(10, 17, "GPE")]}),
#               ("I recently ordered a book from Amazon", {"entities": [(31,37, "ORG")]}),
#               ("I was driving a BMW", {"entities": [(16,19, "PRODUCT")]}),
#               ("I ordered this from ShopClues", {"entities": [(20,29, "ORG")]}),
#               ("Fridge can be ordered in Amazon ", {"entities": [(0,6, "PRODUCT")]}),
#               ("I bought a new Washer", {"entities": [(15,21, "PRODUCT")]}),
#               ("I bought a old table", {"entities": [(15,20, "PRODUCT")]}),
#               ("I bought a fancy dress", {"entities": [(17,22, "PRODUCT")]}),
#               ("I rented a camera", {"entities": [(11,17, "PRODUCT")]}),
#               ("I rented a tent for our trip", {"entities": [(11,15, "PRODUCT")]}),
#               # ("I rented a screwdriver from our neighbour", {"entities": [(11,22, "PRODUCT")]}),
#               # ("I repaired my computer", {"entities": [(14,22, "PRODUCT")]}),
#               # ("I got my clock fixed", {"entities": [(9,14, "PRODUCT")]}),
#               # ("I got my truck fixed", {"entities": [(9,14, "PRODUCT")]}),
#               # ("Flipkart started it's journey from zero", {"entities": [(0,8, "ORG")]}),
#               # ("I recently ordered from Max", {"entities": [(24,27, "ORG")]}),
#               # ("Flipkart is recognized as leader in market",{"entities": [(0,8, "ORG")]}),
#               # ("I recently ordered from Swiggy", {"entities": [(24,30, "ORG")]})
#               ]

TRAIN_DATA=[("3M Health Care ",
            {"entities": [(0, 14, "Health ")]}),
            ("Invoice Number 2001528",
             {"entities": [(97, 104, "Invoice Number")]})]
def train_spacy(data, iterations):
   TRAIN_DATA = data
   nlp = spacy.blank('en')  # create blank Language class
   # create the built-in pipeline components and add them to the pipeline
   # nlp.create_pipe works for built-ins that are registered with spaCy
   if 'ner' not in nlp.pipe_names:
      ner = nlp.create_pipe('ner')
      nlp.add_pipe(ner, last=True)

   # add labels
   for _, annotations in TRAIN_DATA:
      for ent in annotations.get('entities'):
         ner.add_label(ent[2])

   # get names of other pipes to disable them during training
   other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
   with nlp.disable_pipes(*other_pipes):  # only train NER
      optimizer = nlp.begin_training()
      for itn in tqdm(range(iterations)):
         # print("Statring iteration " + str(itn))
         random.shuffle(TRAIN_DATA)
         losses = {}
         for text, annotations in TRAIN_DATA:
            nlp.update(
               [text],  # batch of texts
               [annotations],  # batch of annotations
               drop=0.1,  # dropout - make it harder to memorise data
               sgd=optimizer,  # callable to update weights
               losses=losses)
      print("model created")
   return nlp


start_training = train_spacy(TRAIN_DATA, 50)
start_training.to_disk("spacy_start_model")
# _____________________________________________________________________________________________________
# <remodeling data data data data ********************************************************************>
# _______________________________________________________________________________________
def train_spacy1(data, iterations):  # <-- Add model as nlp parameter
    TRAIN_DATA = data
    # nlp = spacy.load("spacy_start_model1")
    nlp = spacy.load("spacy_start_model")
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        print('getting ner')
        ner = nlp.get_pipe('ner')

    # add labels
    # ner.add_label('Product')
    # ner.add_label('Prema')
    # ner.add_label('life')
    # ner.add_label('person')
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.resume_training()
        for itn in tqdm(range(iterations)):
            #print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                # print(annotations)
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.0,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
                # print(losses)
        print('retrained')
    return nlp


# TRAIN_DATA2 = [ ("Pizza is a common fast food.", {"entities": [(0, 5, "FOOD")]}),
#               ("Pasta is an italian recipe", {"entities": [(0, 5, "FOOD")]}),
#               ("China's noodles are very famous", {"entities": [(8,15, "FOOD")]}),
#               ("Shrimps are famous in China too", {"entities": [(0,7, "FOOD")]}),
#               ("Lasagna is another classic of Italy", {"entities": [(0,7, "FOOD")]}),
#               ("Sushi is extemely famous and expensive Japanese dish", {"entities": [(0,5, "FOOD")]}),
#               ("Unagi is a famous seafood of Japan", {"entities": [(0,5, "FOOD")]}),
#               ("Tempura , Soba are other famous dishes of Japan", {"entities": [(0,7, "FOOD")]}),
#               ("Udon is a healthy type of noodles", {"entities": [(0,4, "ORG")]}),
#               ("Chocolate soufflé is extremely famous french cuisine", {"entities": [(0,17, "FOOD")]}),
#               ("Flamiche is french pastry", {"entities": [(19,25, "FOOD")]}),
#               ("Burgers are the most commonly consumed fastfood", {"entities": [(0,7, "FOOD")]}),
#               ("Burgers are the most commonly consumed fastfood", {"entities": [(0,7, "FOOD")]}),
#               ("Frenchfries are considered too oily", {"entities": [(0,11, "FOOD")]})
#            ]
# #
# TRAIN_DATA_2 = [('Who is a Chala Khan', {"entities": [(9, 14, 'PERSON')]}),
#             ('I like London and Berlin.', {"entities": [(7, 13, 'LOC')]})]
# TRAIN_DATA_3 = [('Naveen is good Person', {"entities": [(0, 6, 'Boy')]}),
#             ('u wanna a manc', {"entities": [(10, 14, 'good')]})]
TRAIN_DATA_3=[("3M Health Care ",{"entities": [(0, 14, "Health "), (97, 104, "Invoice Number"), (133, 141, "Date "), (172, 176, "PO Number")]})]
 # <-- Now your base model is your custom model
# start_training = train_spacy1(TRAIN_DATA_3, 10)
# start_training.to_disk("spacy_start_model1")
# start_training = train_spacy1(TRAIN_DATA_3, 500)
# start_training.to_disk("spacy_start_model3")
#
# import spacy
# nlp = spacy.load("spacy_start_ravinmodel")
# test_text = "what is the price of jeans?"
# doc = nlp(test_text)
# print("Entities in '%s'" %test_text)
# for ent in doc.ents:
#     print(ent.text,ent.label_)



