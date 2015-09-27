from __future__ import division
__author__ = 'dado'

'''

This file creates the model based on the user's taste
1) read the path of images
2) for each picture it detects the face
3) it crops the face out
4) it resize it with a size of 30x30
5) rename the file in <filename>_cropped.jpg
6) convert the pic into a numpy array
7) set the label to either Like or Not Like
8) build the model according to the user selection of features extraction
   and classification method
9) save the model as <username>_model.pkl in the current folder

'''

import os
import sys
import cv2
import numpy as np
import pickle

from enum import Enum
from collections import namedtuple

sys.path.append("../Useful scripts/facerec-master/py")

from facerec.feature import Fisherfaces, PCA, SpatialHistogram, Identity
from facerec.distance import EuclideanDistance, ChiSquareDistance
from facerec.classifier import NearestNeighbor, SVM
from facerec.model import PredictableModel
from facerec.serialization import save_model

from webappflask.models import (
    db,
    Person,
    MeasurementData,
    User,
    Photo,
)
from webappflask.config import photodir

try:
    from PIL import Image
except ImportError:
    import Image

basedir = os.path.abspath(os.path.dirname(__file__))
basedir = os.path.join(basedir, 'webappflask/')

basephotodir = os.path.join(basedir, photodir)

class Feature(Enum):
    pca = PCA()
    spacial = SpatialHistogram()
    identity = Identity()
    fisherfaces = Fisherfaces()

class Classifier(Enum):
    svm = SVM()
    svm_linear = SVM("-s 2 -t 0 -n 0.3 -q") # One-class SVM, linear function, cross-validation k = 3
    svm_rbf = SVM('-s 2 -t 2 -q') # One-class SVM, rbs function, cross-validation k = 3
    svm_sigmoid = SVM('-s 2 -t 3 -n 0.7 -q') # One-class SVM, sigmoid function, nu param cross-validation k = 3

    euclidean = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    chisquare = NearestNeighbor(dist_metric=ChiSquareDistance(), k=1)
    euclidean3 = NearestNeighbor(dist_metric=EuclideanDistance(), k=3)
    chisquare3 = NearestNeighbor(dist_metric=ChiSquareDistance(), k=3)
    euclidean5 = NearestNeighbor(dist_metric=EuclideanDistance(), k=5)
    chisquare5 = NearestNeighbor(dist_metric=ChiSquareDistance(), k=5)
    euclidean7 = NearestNeighbor(dist_metric=EuclideanDistance(), k=7)
    chisquare7 = NearestNeighbor(dist_metric=ChiSquareDistance(), k=7)


def read_images(path):
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                if filename == ".DS_Store":
                    continue

                # detect face
                im = None
                image_path = subject_path + "/" + filename
                cropped_image = detect_face(image_path)
                if cropped_image is not None:
                    im = Image.open(image_path.replace(".jpg", "_cropped.jpg"))
                else:
                    continue

                im = im.convert("L")

                X.append(np.asarray(im, dtype=np.uint8))
                rated = retrieve_rate(image_path.replace("_cropped.jpg", ".jpg"))
                y.append(rated)
    return [X, y]

RatedPerson = namedtuple("RatedPerson", "id pictures like")

def read_images_db(user, setsize=None, photodir=photodir):
    X, y, testpersons = [], [], []
    measurements_query = db.session.query(MeasurementData).join(User).filter(MeasurementData.user == user).subquery()
    persons = db.session.query(Person).outerjoin(measurements_query, measurements_query.c.person_id == Person.id).filter(measurements_query.c.person_id != None).all()
    #print(persons)
    if setsize is not None:
        maxlen = len(persons) / 100 * 60
        #print("persons: {}, maxlen: {}".format(len(persons), maxlen))
    for i, p in enumerate(persons):
        measure = db.session.query(MeasurementData).filter(MeasurementData.person == p).filter(MeasurementData.user == user).first()
        if i <= maxlen:
            for photo in p.photos:
                # detect face
                image_path = os.path.join(basephotodir, photo.filepath)
                cropped_image = detect_face(image_path)

                if cropped_image is not None:
                    with Image.open(image_path.replace(".jpg", "_cropped.jpg")) as im:
                        image = im.convert("L")
                        X.append(np.asarray(image, dtype=np.uint8))
                        if measure.like:
                            rated = 1
                        else:
                            rated = 0
                        y.append(rated)
        else:
            person_photos = []
            for photo in p.photos:
                # detect face
                image_path = os.path.join(basephotodir, photo.filepath)
                cropped_image = detect_face(image_path)

                if cropped_image is not None:
                    with Image.open(image_path.replace(".jpg", "_cropped.jpg")) as im:
                        image = im.convert("L")
                        person_photos.append(np.asarray(image, dtype=np.uint8))
            testpersons.append(RatedPerson(p.id, person_photos, measure.like))



    #print("num X: {}, num y: {}, num testpersons: {}".format(len(X), len(y), len(testpersons)))
    return [X, y], testpersons



def users_db():
    users = db.session.query(User).all()
    for u in users:
        yield u

def detect_face(imagePath):
    # Create the haar cascade
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(imagePath.encode('utf-8'))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = facecascade.detectMultiScale(gray,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    crop_image = None
    i = 0
    for (x, y, w, h) in faces:
        if i > 1:
            return None

        crop_image = image[y:y + h, x:x + w]
        i += 1

    if crop_image is not None:
        filename = imagePath.replace(".jpg", "") + "_cropped.jpg"
        cv2.imwrite(filename.encode('utf-8'), crop_image)
        img = Image.open(filename)
        img.thumbnail((30, 30), Image.ANTIALIAS)
        img.save(filename, "JPEG")

    return crop_image


# retrieve here from the database if the picture has being liked or not
# do your magic here :)
# image_path is the relative path, i.e. photos/Hanna_54354278435y234v52hb34234nj34/0.jpg
def retrieve_rate(image_path):
    return 1


def create_model_file(username, image_path, feature, classifier):
    # read images and set labels
    [X, y] = read_images(image_path)
    # Define the model as the combination
    model = PredictableModel(feature=feature.value, classifier=classifier.value)

    # Compute the Fisherfaces on the given data (in X) and labels (in y):
    model.compute(X, y)

    # We then save the model, which uses Pythons pickle module:
    model_name = username + "_model.pkl"
    save_model(model_name, model)

def create_model_db(user, modelpath, feature, classifier, setsize=None):
    [X, y], testpersons = read_images_db(user, setsize)
    # Define the model as the combination
    model = PredictableModel(feature=feature.value, classifier=classifier.value)

    # Compute the feature-algorithm on the given data (in X) and labels (in y):
    model.compute(X, y)

    # We then save the model, which uses Pythons pickle module:
    model_name = "{}_{}_model.pkl".format(user.username, user.id)
    testpersons_name = "{}_{}_testpersons.pkl".format(user.username, user.id)
    #save_model(os.path.join(modelpath, model_name), model)
    #with open(os.path.join(modelpath, testpersons_name), "w") as picklefile:
    #    pickle.dump(testpersons, picklefile)

    return model, testpersons

'''
USAGE
-----------------------------------------------------------------------------------
arg[1] = username
arg[2] = <path/to/images>
arg[3] = features_extraction -> -P => PCA
                                -S => SpatialHistogram
                                -I => Identity
                                -F => Fisherfaces

arg[4] = classifier --------->  -S => SVM   # Library is missing apparently!
                                            # although I found it here https://github.com/cjlin1/libsvm/tree/master/python
                                -ND => Nearest Neighbour with Euclidean Distance
                                -NC => Nearest Neighbour with ChiSquare Distance
-----------------------------------------------------------------------------------
'''

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print "USAGE: model_creator.py username </path/to/images> <feature> <classifier>"
        sys.exit()

    # save username for later
    username = sys.argv[1]

    # Now read in the image data. This must be a valid path!
    image_path = sys.argv[2]
    if not os.access(image_path, os.W_OK):
        print "Path is invalid or does not exist"
        sys.exit()

    # set feature extraction
    feature_extraction = sys.argv[3]
    if feature_extraction == "-P":
        # Define the PCA as Feature Extraction method:
        feature = Feature.pca
    elif feature_extraction == "-S":
        # Define the SpatialHistogram as Feature Extraction method:
        feature = Feature.spacial
    elif feature_extraction == "-I":
        # Simplest AbstractFeature you could imagine. It only forwards the data and does not operate on it,
        # probably useful for learning a Support Vector Machine on raw data for example!
        feature = Feature.identity
    elif feature_extraction == "-F":
        # Define the Fisherfaces as Feature Extraction method:
        feature = Feature.fisherfaces
    else:
        print "USAGE: select the correct feature extraction (-P, -S, -I, -F)"
        sys.exit()

    # set classifier method
    classifier_method = sys.argv[4]
    if classifier_method == "-S":
        # Support Vector Machine with default parameters
        classifier = Classifier.svm
    elif classifier_method == "-ND":
        # Define a 1-NN classifier with Euclidean Distance:
        classifier = Classifier.euclidean
    elif classifier_method == "-NC":
        # Define a 1-NN classifier with ChiSquare Distance:
        classifier = Classifier.chisquare
    else:
        print "USAGE: Use the correct classifier method (-S, -ND, -NC)"
        sys.exit()

    create_model_file(username, image_path, feature, classifier)
