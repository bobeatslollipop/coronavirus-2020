import sys
sys.path.insert(0, '..')

from utils import data
import os, csv
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def knn_diff():
    # ------------ HYPERPARAMETERS -------------
    BASE_PATH = 'COVID-19/csse_covid_19_data/'
    N_NEIGHBORS = 5
    MIN_CASES = 1000
    NORMALIZE = True
    # ------------------------------------------

    confirmed = os.path.join(
        BASE_PATH, 
        'csse_covid_19_time_series',
        'time_series_covid19_confirmed_global.csv')
    confirmed = data.load_csv_data(confirmed)
    features = []
    targets = []

    for val in np.unique(confirmed["Country/Region"]):
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)
        features.append(cases)
        targets.append(labels)

    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    predictions = {}

    for _dist in ['minkowski', 'manhattan']:
        for val in np.unique(confirmed["Country/Region"]):
            # test data
            df = data.filter_by_attribute(
                confirmed, "Country/Region", val)
            cases, labels = data.get_cases_chronologically(df)

            # filter the rest of the data to get rid of the country we are
            # trying to predict
            mask = targets[:, 1] != val
            tr_features = features[mask]
            tr_targets = targets[mask][:, 1]

            above_min_cases = tr_features.sum(axis=-1) > MIN_CASES
            tr_features = np.diff(tr_features[above_min_cases], axis=-1)
            if NORMALIZE:
                tr_features = tr_features / tr_features.sum(axis=-1, keepdims=True)

            tr_targets = tr_targets[above_min_cases]

            # train knn
            knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric=_dist)
            knn.fit(tr_features, tr_targets)

            # predict
            cases = np.diff(cases.sum(axis=0, keepdims=True), axis=-1)
            # nearest country to this one based on trajectory
            label = knn.predict(cases)
            
            if val not in predictions:
                predictions[val] = {}
            predictions[val][_dist] = label.tolist()

    with open('results/knn_diff.json', 'w') as f:
        json.dump(predictions, f, indent=4)


def Mixture():
    # ------------ HYPERPARAMETERS -------------
    BASE_PATH = 'COVID-19/csse_covid_19_data/'
    N_NEIGHBORS = 5
    MIN_CASES = 1000
    NORMALIZE = True
    # ------------------------------------------

    confirmed = os.path.join(
        BASE_PATH, 
        'csse_covid_19_time_series',
        'time_series_covid19_confirmed_global.csv')
    confirmed = data.load_csv_data(confirmed)
    features = []
    targets = []
    predictions = {}

    for val in np.unique(confirmed["Country/Region"]):
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)
        features.append(cases)
        targets.append(labels)

    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    if NORMALIZE:
            features = features / features.sum(axis=-1, keepdims=True)

    for n in range(3, 4):
        learner = KMeans(n_clusters=n)
        labels = learner.fit_predict(features)
        unique_labels = np.unique(labels)
        for val in unique_labels:
            predictions[int(val)] = np.unique(targets[labels == val][:, 1])
        with open(f'results/gmm_{n}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(3):
                writer.writerow(predictions[i])
                writer.writerow([])
        print(n, ' ', silhouette_score(features, labels, metric='euclidean'))
        

Mixture()
