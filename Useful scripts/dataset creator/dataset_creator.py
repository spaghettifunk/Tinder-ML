__author__ = 'dado'

import sys
import os
import cv2
import shutil

sys.path.append("../..")
# import facerec modules
from facerec.feature import Fisherfaces, SpatialHistogram, Identity
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
detectAndCropImages = True
saveCroppedImages = True
maxPicturesAllowed = 3

def fix_folders(path):
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            files = os.listdir(subject_path)
            for filename in os.listdir(subject_path):
                if "_cropped.jpg" not in filename:
                    temp = filename
                    number = temp.replace(".jpg", "_cropped.jpg")
                    if number not in files:
                        os.remove(subject_path + "/" + filename)
                else:
                    temp = filename
                    number = temp.replace("_cropped.jpg", ".jpg")
                    if number not in files:
                        os.remove(subject_path + "/" + filename)

def fix_dataset(path):
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            length = len(os.listdir(subject_path))

            if length < (maxPicturesAllowed * 2):   # "* 2" because we both have cropped and normal pics
                shutil.rmtree(subject_path)
            elif length > (maxPicturesAllowed * 2):
                while length != (maxPicturesAllowed * 2):
                    files = os.listdir(subject_path)
                    os.remove(subject_path + "/" + files[0])
                    length = len(os.listdir(subject_path))


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
                    logger.info("Reading image: " + subject_path + "/" + filename)

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
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30),
                                         flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    crop_image = None
    i = 0
    for (x, y, w, h) in faces:
        if i > 1:
            return None

        roi_color = image[y:y + h, x:x + w]
        crop_image = roi_color  # crop image
        i += 1

    if crop_image is not None:
        if save == True:
            cv2.imwrite(imagePath.replace(".jpg", "") + "_cropped.jpg", crop_image)
            # logger.info("Cropped image: " + imagePath)

    return crop_image


if __name__ == "__main__":
    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) < 2:
        print "USAGE: dataset_creator.py </path/to/images>"
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
    read_images(sys.argv[1], (400, 400), detectAndCropImages, saveCroppedImages)
    fix_folders(sys.argv[1])
    fix_dataset(sys.argv[1])