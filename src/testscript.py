from __future__ import division
__author__ = 'tobias'

import os
import sys
import webappflask.models as m
import model_creator as mc
import pickle
import csv
import pygal

from facerec.serialization import load_model

generate_new = True

foldername = 'Identity_1NN_Euclidean'
fieldnames = ['username', 'guessed_correct', 'false-positive', 'false-negative']

modes = [
    # ('PCA_1NN_Euclidean', mc.Feature.pca, mc.Classifier.euclidean),
    # ('PCA_3NN_Euclidean', mc.Feature.pca, mc.Classifier.euclidean3),
    # ('PCA_5NN_Euclidean', mc.Feature.pca, mc.Classifier.euclidean5),
    # ('PCA_7NN_Euclidean', mc.Feature.pca, mc.Classifier.euclidean7),
    #
    # ('Fisherface_1NN_Euclidean', mc.Feature.fisherfaces, mc.Classifier.euclidean),
    # ('Fisherface_3NN_Euclidean', mc.Feature.fisherfaces, mc.Classifier.euclidean3),
    # ('Fisherface_5NN_Euclidean', mc.Feature.fisherfaces, mc.Classifier.euclidean5),
    # ('Fisherface_7NN_Euclidean', mc.Feature.fisherfaces, mc.Classifier.euclidean7),
    #
    # ('PCA_1NN_Chisquare', mc.Feature.pca, mc.Classifier.chisquare),
    # ('PCA_3NN_Chisquare', mc.Feature.pca, mc.Classifier.chisquare3),
    # ('PCA_5NN_Chisquare', mc.Feature.pca, mc.Classifier.chisquare5),
    # ('PCA_7NN_Chisquare', mc.Feature.pca, mc.Classifier.chisquare7),
    #
    # ('Fisherface_1NN_Chisquare', mc.Feature.fisherfaces, mc.Classifier.chisquare),
    # ('Fisherface_3NN_Chisquare', mc.Feature.fisherfaces, mc.Classifier.chisquare3),
    # ('Fisherface_5NN_Chisquare', mc.Feature.fisherfaces, mc.Classifier.chisquare5),
    # ('Fisherface_7NN_Chisquare', mc.Feature.fisherfaces, mc.Classifier.chisquare7),
    #
    # ('Spacial_1NN_Euclidean', mc.Feature.spacial, mc.Classifier.euclidean),
    # ('Spacial_3NN_Euclidean', mc.Feature.spacial, mc.Classifier.euclidean3),
    # ('Spacial_5NN_Euclidean', mc.Feature.spacial, mc.Classifier.euclidean5),
    # ('Spacial_7NN_Euclidean', mc.Feature.spacial, mc.Classifier.euclidean7),
    #
    # ('Spacial_1NN_Chisquare', mc.Feature.spacial, mc.Classifier.chisquare),
    # ('Spacial_3NN_Chisquare', mc.Feature.spacial, mc.Classifier.chisquare3),
    # ('Spacial_5NN_Chisquare', mc.Feature.spacial, mc.Classifier.chisquare5),
    # ('Spacial_7NN_Chisquare', mc.Feature.spacial, mc.Classifier.chisquare7),
    #
    # ('Identity_1NN_Euclidean', mc.Feature.identity, mc.Classifier.euclidean),
    # ('Identity_3NN_Euclidean', mc.Feature.identity, mc.Classifier.euclidean3),
    # ('Identity_5NN_Euclidean', mc.Feature.identity, mc.Classifier.euclidean5),
    # ('Identity_7NN_Euclidean', mc.Feature.identity, mc.Classifier.euclidean7),
    #
    # ('Identity_1NN_Chisquare', mc.Feature.identity, mc.Classifier.chisquare),
    # ('Identity_3NN_Chisquare', mc.Feature.identity, mc.Classifier.chisquare3),
    # ('Identity_5NN_Chisquare', mc.Feature.identity, mc.Classifier.chisquare5),
    # ('Identity_7NN_Chisquare', mc.Feature.identity, mc.Classifier.chisquare7),

    ('Identity_SVM', mc.Feature.identity, mc.Classifier.svm),
    # ('PCA_SVM', mc.Feature.pca, mc.Classifier.svm),
    # ('Fisherfaces_SVM', mc.Feature.fisherfaces , mc.Classifier.svm),
    # ('Spatial_SVM', mc.Feature.spacial, mc.Classifier.svm)

    # ('Identity_SVM_Linear', mc.Feature.identity, mc.Classifier.svm_linear),
    # ('PCA_SVM_Linear', mc.Feature.pca, mc.Classifier.svm_linear),
    # ('Fisherfaces_SVM_Linear', mc.Feature.fisherfaces , mc.Classifier.svm_linear),
    # ('Spatial_SVM_Linear', mc.Feature.spacial, mc.Classifier.svm_linear)

    # ('Identity_SVM_RBF', mc.Feature.identity, mc.Classifier.svm_rbf),
    # ('PCA_SVM_RBF', mc.Feature.pca, mc.Classifier.svm_rbf),
    # ('Fisherfaces_SVM_RBF', mc.Feature.fisherfaces , mc.Classifier.svm_rbf),
    # ('Spatial_SVM_RBF', mc.Feature.spacial, mc.Classifier.svm_rbf)

    # ('Identity_SVM_RBF', mc.Feature.identity, mc.Classifier.svm_sigmoid),
    # ('PCA_SVM_RBF', mc.Feature.pca, mc.Classifier.svm_sigmoid),
    # ('Fisherfaces_SVM_RBF', mc.Feature.fisherfaces , mc.Classifier.svm_sigmoid),
    # ('Spatial_SVM_RBF', mc.Feature.spacial, mc.Classifier.svm_sigmoid)
]

def test_on_users(testuser, writer_nonvote, writer_vote, feature, classifier, foldername):
    print(testuser)

    modelpath = 'models/{}/{}_{}'.format(foldername, testuser.username, testuser.id)
    if not os.path.exists(modelpath):
    #if True:
        os.makedirs(modelpath)

    model_name = "{}_{}_model.pkl".format(testuser.username, testuser.id)
    testpersons_name = "{}_{}_testpersons.pkl".format(testuser.username, testuser.id)

    if generate_new or not os.path.exists(os.path.join(modelpath, model_name)):
        model, testpersons_pickle = mc.create_model_db(testuser, modelpath, feature, classifier, setsize=60)

    #model = load_model(os.path.join(modelpath, model_name))

    #with open(os.path.join(modelpath, testpersons_name), "r") as picklefile:
    #        testpersons_pickle = pickle.load(picklefile)

    # assert(len(testtuple[0]) == len(testtuple_pickle[0]))
    # assert(len(testtuple[1]) == len(testtuple_pickle[1]))

    result_vote = []
    result_nonvote = []
    for p in testpersons_pickle:
        likes = 0
        for picture in p.pictures:

            interm = model.predict(picture)
            print(interm)
            [predict, _] = interm
            if predict == 1:
                likes += 1
            result_nonvote.append((p.like, predict))
        if likes > len(p.pictures) / 2:
            result_vote.append((p.like, 1))
        else:
            result_vote.append((p.like, 0))


    correct_vote, false_negative_vote, false_positive_vote = 0, 0, 0
    print(result_vote)
    for (like, predict) in result_vote:
        if like == predict:
            correct_vote += 1
        elif like == 1 and predict == 0:
            false_negative_vote += 1
        else:
            false_positive_vote += 1

    writer_vote.writerow({'username': testuser.username, 'guessed_correct': correct_vote, 'false-positive': false_positive_vote, 'false-negative': false_negative_vote})

    print("VOTED: user: {}, guessed correctly: {}, false positive: {}, false negative: {}".format(testuser, correct_vote, false_positive_vote, false_negative_vote))

    print(result_nonvote)
    correct_nonvote, false_negative_nonvote, false_positive_nonvote = 0, 0, 0
    for (like, predict) in result_nonvote:
        if like == predict:
            correct_nonvote += 1
        elif like == 1 and predict == 0:
            false_negative_nonvote += 1
        else:
            false_positive_nonvote += 1

    writer_nonvote.writerow({'username': testuser.username, 'guessed_correct': correct_nonvote, 'false-positive': false_positive_nonvote, 'false-negative': false_negative_nonvote})

    print("NONVOTED: user: {}, guessed correctly: {}, false positive: {}, false negative: {}".format(testuser, correct_nonvote, false_positive_nonvote, false_negative_nonvote))

def create_svg(foldername, ending=''):
    usernames,correct, false_pos, false_neg = [], [], [], []
    with open(os.path.join('results', foldername + ending + '.csv'), 'rb') as csvf:
        reader = csv.DictReader(csvf, delimiter=';')

        for row in reader:
            usernames.append(row[fieldnames[0]])
            correct.append(int(row[fieldnames[1]]))
            false_pos.append(int(row[fieldnames[2]]))
            false_neg.append(int(row[fieldnames[3]]))


    stackedbar_chart = pygal.StackedBar()
    stackedbar_chart.title = foldername
    stackedbar_chart.x_labels = usernames
    stackedbar_chart.add('Correct', correct)
    stackedbar_chart.add('false pos', false_pos)
    stackedbar_chart.add('false neg', false_neg)
    stackedbar_chart.render_to_file(os.path.join('results', foldername + ending + '.svg'))


def test_all_users(foldername, feature, classifier):
    with open(os.path.join('results', foldername + '_vote.csv'), 'wb') as csv_vote:
        with open(os.path.join('results', foldername + '_nonvote.csv'), 'wb') as csv_nonvote:
            writer_nonvote = csv.DictWriter(csv_nonvote, fieldnames=fieldnames, delimiter=';')
            writer_nonvote.writeheader()
            writer_vote = csv.DictWriter(csv_vote, fieldnames=fieldnames, delimiter=';')
            writer_vote.writeheader()

            for user in m.db.session.query(m.User).all():
                test_on_users(user, writer_nonvote=writer_nonvote, writer_vote=writer_vote, feature=feature, classifier=classifier, foldername=foldername)
                csv_nonvote.flush()
                csv_vote.flush()

    create_svg(foldername, ending='_vote')
    create_svg(foldername, ending='_nonvote')

if __name__ == '__main__':
    for i in modes:
        test_all_users(i[0], i[1], i[2])
