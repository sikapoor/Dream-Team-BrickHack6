import os
import urllib.request
from app import app
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename

# Importing all libraries for vision related computation
import keras
from flask import Flask
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
# Importing libraries for NLP related computation
import gensim
from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import spacy
import itertools


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# Loading dataset for creating embeddings
nlp = spacy.load('en_core_web_md')

# Hardcoded strings ====> objects not allowed in carry on during TSA pre check
prohibited = ["Explosive and incendiary materials", "deodrant", "Flammable",
              "Gasses and pressure containers", "fire Matchstick peroxides",
              "Poison", "Infection", "Corrosive", "Organics", "Radioactive",
              "Magnetic", "magnet", "Marijuana", "cannabis", "Stick", "golf club",
              "sports", "Firearms", "ammunition", "Knives", "cutting instruments",
              "sharp object", "Paintball", "guns", "Power tool", "hand tool", "Dry ice",
              "hatchet", "axe", "sickle", "scythe", "pitchfork", "spade", "shovel", "trowel",
              "hoe", "fork", "rake", "alcohol", "beer", "wine", "bottle", "Arc Lighters", "Plasma Lighters",
              "Electronic Lighters", "Lighter", "burner", "BB Gun", "Cap Guns", "Compressed Air Gun",
              "Flare Guns", "Gun Powder", "Holsters", "Rifle", "Rocket Launcher",
              "Pistol", "Cattle Prod", "Cooking spray",
              "Crowbars", "Drills", "Drill Bits", "Engine", "Fuel",
              "Engine-powered Equipment with Residual Fuel", "Heating Pad Gel",
              "Mallet", "Microwave", "Nail Guns", "Galaxy", "Screwdriver"]


def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


def classify(cosine_arr):
    if max(cosine_arr) > 0.42:
        return True
    else:
        return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/file-upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        mobile = keras.applications.mobilenet.MobileNet()
        # Getting predicition from the model
        preprocessed_image = prepare_image(
            os.path.join(app.config['UPLOAD_FOLDER'], filename))
        predictions = mobile.predict(preprocessed_image)
        results = imagenet_utils.decode_predictions(predictions)
        # results is a nested list of tuples with predictions

        # mini script to convert predictions to more useful words for embedding creation
        pred_arr = []
        for i in results[0]:
            pred_arr.append(i[1].split("_"))

        # Flattens the nested list formatted predictions
        new_d = [i[0] for i in pred_arr]

        # Finding cosine similarity between all possible combinations
        cosine_arr = []
        for i in new_d:
            for j in prohibited:
                token1 = nlp(i)
                token2 = nlp(j)
                cosine_arr.append(token1.similarity(token2))

        isNotAllowed = classify(cosine_arr)
        if isNotAllowed:
            resp = jsonify({'message': 'This item is not allowed'})
        else:
            resp = jsonify({'message': 'This item is allowed'})

        resp.status_code = 201
        return resp
    else:
        resp = jsonify(
            {'message': 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    app.run(port=5001)
