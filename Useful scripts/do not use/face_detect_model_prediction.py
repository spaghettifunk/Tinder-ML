#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Philipp Wagner <bytefish[at]gmx[dot]de>.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the author nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys
import os
import cv2

sys.path.append("../..")
# import facerec modules
from facerec.feature import Fisherfaces, PCA, SpatialHistogram, Identity
from facerec.distance import EuclideanDistance, ChiSquareDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
from facerec.serialization import save_model, load_model
# import numpy, matplotlib and logging
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import matplotlib.cm as cm
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from facerec.lbp import LPQ, ExtendedLBP

# debugging variables
detectAndCropImages = False
saveCroppedImages = False

def read_images(path, sz=None, detectAndCrop=True, save=True):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if filename == ".DS_Store":
                        continue

                    # detect face
                    im = None
                    if detectAndCrop:
                        cropped_image = detect_face(subject_path + "/" + filename, save)
                        if cropped_image is not None:
                            im = Image.open(os.path.join(subject_path, filename.replace(".jpg", "_cropped.jpg")))
                        else:
                            continue

                    # avoid pictures that are not cropped because it means that
                    # the algorithm didn't detect a face
                    if "_cropped.jpg" not in filename:
                        continue
                    elif im is None:  # we already cropped the image, so let's load it
                        im = Image.open(os.path.join(subject_path, filename))

                    im = im.convert("L")
                    # resize to given size (if given)
                    if sz is not None:
                        im = im.resize(sz, Image.ANTIALIAS)

                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)

                    logger.info("Loading image: " + subject_path + "/" + filename)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c += 1
    return [X, y]

def detect_face(imagePath, save):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    crop_image = None
    i = 0
    for (x, y, w, h) in faces:
        if i > 1:
            return None

        roi_color = image[y:y + h, x:x + w]
        crop_image = roi_color    # crop image
        i += 1

    if crop_image is not None:
        if save == True:
            cv2.imwrite(imagePath.replace(".jpg", "") + "_cropped.jpg", crop_image)
            #logger.info("Cropped image: " + imagePath)

    return crop_image

if __name__ == "__main__":
    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) < 2:
        print "USAGE: face_detect_model_prediction.py </path/to/images>"
        sys.exit()

    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # Now read in the image data. This must be a valid path!
    [X, y] = read_images(sys.argv[1], (400, 400), detectAndCropImages, saveCroppedImages)

    # Define the PCA as Feature Extraction method:
    #feature = PCA()

    # Define the Fisherfaces as Feature Extraction method:
    feature = Fisherfaces()

    # Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)

    # Define the model as the combination
    my_model = PredictableModel(feature=feature, classifier=classifier)

    # Compute the Fisherfaces on the given data (in X) and labels (in y):
    my_model.compute(X, y)

    # We then save the model, which uses Pythons pickle module:
    save_model('model_like.pkl', my_model)
    #model = load_model('model.pkl')

    # Then turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    '''
    E = []
    for i in xrange(min(model.feature.eigenvectors.shape[1], 122)):
        e = model.feature.eigenvectors[:, i].reshape(X[0].shape)
        E.append(minmax_normalize(e, 0, 255, dtype=np.uint8))

    # Plot them and store the plot to "python_fisherfaces_fisherfaces.pdf"
    subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet, filename="fisherfaces.png")

    # Perform a 10-fold cross validation
    cv = KFoldCrossValidation(model, k=10)
    cv.validate(X, y)

    # And print the result:
    cv.print_results()
    '''
