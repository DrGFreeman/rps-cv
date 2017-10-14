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

import numpy as np

# Settings:

# Classifier output .pkl filename
pklFilename = 'clf.pkl'

# Stratified KFold cross-validation parameter
n_splits = 8

# Grid Search parameters
pca__n_components = [40]
clf__gamma = np.logspace(-4, -2, 5)
clf__C = np.logspace(.5, 2, 5)
scoring = 'f1_macro'
n_jobs = 4


def train():
    import time
    t0 = time.time()

    def dt():
        return round(time.time() - t0, 2)

    print('+{}: Importing libraries'.format(dt()))

    import os
    from glob import glob
    import pickle

    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    import rpsimgproc as imp
    import rpsutil as rps

    # Generate image data from stored images
    print('+{}: Generating image data'.format(dt()))
    features, labels = imp.generateGrayFeatures(verbose=False)

    unique, count = np.unique(labels, return_counts=True)

    # Print the number of traning images for each label
    for i, label in enumerate(unique):
        print('{}: {} images'.format(rps.gestureTxt[label], count[i]))

    # Define pipeline parameters
    print('+{}: Defining pipeline'.format(dt()))
    steps = [('pca', PCA()), ('clf', SVC(kernel='rbf'))]
    pipe = Pipeline(steps)

    # Define cross-validation parameters
    print('+{}: Defining cross-validation'.format(dt()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

    # Define grid-search parameters
    print('+{}: Defining grid search'.format(dt()))
    grid_params = dict(pca__n_components=pca__n_components,
                       clf__gamma=clf__gamma,
                       clf__C=clf__C)
    grid = GridSearchCV(pipe, grid_params, scoring=scoring, n_jobs=n_jobs, cv=cv)
    print(grid)

    # Fit the classifier
    print('+{}: Fitting classifier'.format(dt()))
    grid.fit(features, labels)

    # Print the best score and best parameters from the grid-search
    print(grid.best_score_)
    print(grid.best_params_)

    # Write classifier to a .pkl file
    print('+{}: Writing classifier to {}'.format(dt(), pklFilename))
    f = open(pklFilename, 'wb')
    f.flush()
    pickle.dump(grid, f)
    f.close()

    print('+{}: Done!'.format(dt()))

if __name__ == '__main__':
    train()
