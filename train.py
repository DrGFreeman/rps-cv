# train.py
# Source: https://github.com/DrGFreeman/rps-cv
#
# MIT License
#
# Copyright (c) 2017 Julien de la Bruere-Terreault <drgfreeman@tuta.io>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This script reads the pre-processed image data and trains the image
# classifier. The trained classifier is stored in a .pkl (pickle) file.

import sys
import numpy as np

# Settings:

# Random State
rs = 42

# Classifier output .pkl filename
pklFilename = 'clf.pkl'

# Stratified KFold cross-validation parameter
n_splits = 5

# Grid Search parameters
pca__n_components = [60]
#pca__n_components = [40, 60, 80]
#clf__gamma = np.logspace(-4, -3, 3)
clf__gamma = np.logspace(-4, -2, 5)
#clf__C = np.logspace(0, 1, 3)
clf__C = np.logspace(0, 2, 5)
scoring = 'f1_micro'
n_jobs = 4


def train(nbImg=0, cvScore=False):
    import time
    t0 = time.time()

    def dt():
        return round(time.time() - t0, 2)

    print('+{}: Importing libraries'.format(dt()))

    import pickle

    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    import rpsimgproc as imp
    import rpsutil as rps

    # Generate image data from stored images
    print('+{}: Generating image data'.format(dt()))
    features, labels = imp.generateGrayFeatures(nbImg=nbImg, verbose=False,
                                                rs=rs)

    unique, count = np.unique(labels, return_counts=True)

    # Print the number of traning images for each label
    for i, label in enumerate(unique):
        print('  {}: {} images'.format(rps.gestureTxt[label], count[i]))

    # Generate test set
    print('+{}: Generating test set'.format(dt()))
    sssplit = StratifiedShuffleSplit(n_splits=1, test_size=.15, random_state=rs)
    for train_index, test_index in sssplit.split(features, labels):
        features_train = features[train_index]
        features_test = features[test_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]

    # Define pipeline parameters
    print('+{}: Defining pipeline'.format(dt()))
    steps = [('pca', PCA()), ('clf', SVC(kernel='rbf'))]
    pipe = Pipeline(steps)

    # Define cross-validation parameters
    print('+{}: Defining cross-validation'.format(dt()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)

    # Define grid-search parameters
    print('+{}: Defining grid search'.format(dt()))
    grid_params = dict(pca__n_components=pca__n_components,
                       clf__gamma=clf__gamma,
                       clf__C=clf__C)
    grid = GridSearchCV(pipe, grid_params, scoring=scoring, n_jobs=n_jobs,
        refit=True, cv=cv, verbose=1)
    print('Grid search parameters:')
    print(grid)

    # Fit the classifier
    t0_train = time.time()
    print('+{}: Fitting classifier'.format(dt()))
    grid.fit(features_train, labels_train)
    dt_train = time.time() - t0_train

    if cvScore:
        # Print the results of the grid search cross-validation
        cvres = grid.cv_results_
        print('Cross-validation results:')
        for score, std, params in zip(cvres['mean_test_score'],
                cvres['std_test_score'], cvres['params']):
            print('  {}, {}, {}'.format(round(score, 4), round(std, 5), params))

    # Print the best score and best parameters from the grid-search
    print('Grid search best score: {}'.format(grid.best_score_))
    print('Grid search best parameters:')
    for key, value in grid.best_params_.items():
        print('  {}: {}'.format(key, value))

    # Validate classifier on test set
    print('+{}: Validating classifier on test set'.format(dt()))
    pred = grid.predict(features_test)
    score = f1_score(labels_test, pred, average='micro')
    print('Classifier f1-score on test set: {}'.format(score))
    print('Confusion matrix:')
    print(confusion_matrix(labels_test, pred))
    print('Classification report:')
    tn = [rps.gestureTxt[i] for i in range(1, 4)]
    print(classification_report(labels_test, pred, target_names=tn))

    # Write classifier to a .pkl file
    print('+{}: Writing classifier to {}'.format(dt(), pklFilename))
    with open(pklFilename, 'wb') as f:
        f.flush()
        pickle.dump(grid, f)

    print('+{}: Done!'.format(dt()))

    return grid.best_score_, score, dt_train

if __name__ == '__main__':

    # Read command line arguments
    argv = sys.argv

    cvScore = False

    if len(sys.argv) > 1:
        for arg in argv[1:]:
            if arg == 'cvScore':
                cvScore = True

    train(cvScore)
