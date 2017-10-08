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

# Images input .csv filename prefix
csvFilename = 'gray'

# Classifier output .pkl filename
pklFilename = 'clf.pkl'

# Stratified KFold cross-validation parameter
n_splits = 8

# Grid Search parameters
pca__n_components = [20]
clf__gamma = np.logspace(-5, -3, 5)
clf__C = np.logspace(.5, 2, 4)
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

    import pandas as pd

    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    # Create empty dataframe to store image data
    data = pd.DataFrame()

    # Create list of .csv files
    files = glob(os.path.join('*.csv'))
    files.sort(key=str.lower)

    for filename in files:
        print('+{}: Loading data from {}'.format(dt(), filename))
        dataLoad = pd.DataFrame.from_csv(filename)
        data = data.append(dataLoad, ignore_index=True)

    data.info()

    print(data.label.value_counts().sort_index())

    #data = data.sample(frac=1.)

    print('+{}: Splitting labels and features'.format(dt()))
    labels = data.label
    features = data.drop(['label', 'path'], axis=1)

    print('+{}: Defining pipeline'.format(dt()))

    steps = [('pca', PCA()), ('clf', SVC(kernel='rbf'))]
    pipe = Pipeline(steps)

    print('+{}: Defining cross-validation'.format(dt()))

    cv = StratifiedKFold(n_splits=n_splits)

    print('+{}: Defining grid search'.format(dt()))

    grid_params = dict(pca__n_components=pca__n_components,
                       clf__gamma=clf__gamma,
                       clf__C=clf__C)
    grid = GridSearchCV(pipe, grid_params, scoring=scoring, n_jobs=n_jobs, cv=cv)

    print('+{}: Fitting classifier'.format(dt()))

    grid.fit(features, labels)

    print(grid.best_score_)
    print(grid.best_params_)

    print('+{}: Writing classifier to {}'.format(dt(), pklFilename))

    f = open(pklFilename, 'wb')
    f.flush()
    pickle.dump(grid, f)
    f.close()

    print('+{}: Done!'.format(dt()))

if __name__ == '__main__':
    train()
