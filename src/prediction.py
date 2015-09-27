__author__ = 'dado'

'''

This file apply the predicition on a test dataset
1) get the user's model
2) get the test_images
3) for each person

'''

import sys
import os

sys.path.append("../..")
from facerec.serialization import load_model
import numpy as np
from PIL import Image

def read_images_file(path, model):
    matches = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)

            # save the person's name
            rated_person = subdirname   # is it better the whole path instead ?

            imageMatrix = []
            label = []
            predicted_labels = []
            distances = []
            for filename in os.listdir(subject_path):
                if (filename == ".DS_Store") or ("_cropped.jpg" not in filename):
                    continue

                image_path = subject_path + "/" + filename
                im = Image.open(image_path)
                im = im.convert("L")

                imageMatrix.append(np.asarray(im, dtype=np.uint8))

                # rated = retrieve_rate(image_path)
                # label.append(rated)

                # get prediction
                # print imageMatrix[0]
                prediction = model.predict(imageMatrix[0])

                # save predicted label
                predicted_labels.append(prediction[0])

                # get the classifier output for calculating the distance
                classifier_output = prediction[1]

                # Need to modify this line based on the model we previously created
                # -----------------------------------------------------------------
                # Now let's get the distance from the assuming a 1-Nearest Neighbor.
                # Since it's a 1-Nearest Neighbor only look take the zero-th element:
                distances.append(classifier_output['distances'][0])

            # let's get some results
            result_prediction = model_prediction(rated_person, predicted_labels, distances)
            matches.append(result_prediction)

    # return all the predictions
    return matches

def model_prediction(rated_person, predicted_labels, distances):
        # we have all the distances and the predicted label
        # we can zip them together to get something like this
        # [(predicted_label_pic1, distance_pic1) , (predicted_label_pic2, distance_pic2), ...]
        combined = zip(predicted_labels, distances)

        # at this point we should average these values to claim that
        # this person can be part of the Like or Not Like
        transformed = np.array(combined)
        result_average = np.mean(transformed, axis=0)

        # result_average contains the final result
        # we should save these results in a structure
        return [rated_person, result_average]

# retrieve here from the database if the picture has being liked or not
# do your magic here :)
# image_path is the relative path, i.e. photos/Hanna_54354278435y234v52hb34234nj34/0.jpg
def retrieve_rate(image_path):
    return 1


'''
USAGE
-----------------------------------------------------------------------------------
arg[1] = user_model         # without ".pkl"
arg[2] = <path/to/images>
-----------------------------------------------------------------------------------
'''
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "USAGE: test_prediction.py username </path/to/images> <feature> <classifier>"
        sys.exit()

    user_model = sys.argv[1]
    test_images_path = sys.argv[2]

    model = load_model(user_model + ".pkl")
    matches = read_images(test_images_path, model)

    # We should plot something and
    # create a folder with Liked people and Not Liked people
