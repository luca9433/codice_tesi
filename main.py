# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 23:54:50 2021

@author: Admin
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
from sklearn.svm import SVC
import umap.umap_ as umap
import cv2

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def umap_proj_plot(data,classes,labels,save_path=None):#creates a 2D UMAP projection of a dataset containing multidimensional vectors, for example PIs, divided into a certain 
    reducer = umap.UMAP()                              #number of classes
    reducer.fit(data)
    projector = reducer.transform(data)
    f, ax = plt.subplots(figsize=(10,10))
    for i, genre in enumerate(classes):
        inds = np.where(np.asarray(labels) == genre)[0]
        colors = [sns.color_palette("colorblind")[i] for _ in inds]
        ax.scatter(projector[inds, 0], 
                   projector[inds, 1], 
                   c = colors, 
                   label=genre,
                   alpha=.3)
    leg = plt.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)    
    if save_path is not None:
        f.savefig(os.path.join(save_path, "umap_genres.svg"))
        f.savefig(os.path.join(save_path, "umap_genres.png"))
    return projector

def cross_validation(pipeline, X, y, n_folds=5, random_state=None):#performs a cross-validation of a ML pipeline, splitting the dataset with StaratifiedKFold, 
                                                                     #returning accuracy scores in train and test and confusion matrices
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=False) 
    cms, accs = [], []
    cms_train, accs_train = [], []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = make_pipeline(pipeline)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cms.append(confusion_matrix(y_test, y_pred)) 
        accs.append(accuracy_score(y_test, y_pred)) 
        y_pred_train = clf.predict(X_train)
        cms_train.append(confusion_matrix(y_train, y_pred_train))
        accs_train.append(accuracy_score(y_train, y_pred_train))
        
    return cms, accs, cms_train, accs_train


def plot_accuracy(accs_test, accs_train, ax = None, save_plot=None, plot_name=None):#draws a barplot where we can visualise accuracy scores in train and test
    if ax is None:
        f, ax = plt.subplots()
        
    assert isinstance(accs_test, list) and isinstance(accs_train, list), "Please, provide lists as input instead of {}".format(type(accs_test))
    df = pd.DataFrame({"acc": accs_test + accs_train, "is_test": ["test" for _ in accs_test] + ["train" for _ in accs_train]})
    sns.barplot(x="is_test", y="acc", data=df, ax=ax)
    if save_plot is not None and plot_name is not None:
        f.savefig(os.path.join(save_plot, plot_name+"accuracy_barplot.svg"))
        f.savefig(os.path.join(save_plot, plot_name+"accuracy_barplot.png"))
    return df

def main(path_to_data_folder="C:\\Users\\Admin\\Documents\\python"):
    
    mfccs_paths=[os.path.join(path_to_data_folder, f) 
                               for f in os.listdir(path_to_data_folder) 
                               if ".npy" in f and "PI" not in f]
    genres = ["blues", "classical", "country", "disco", "hiphop", 
              "jazz", "metal", "pop", "reggae", "rock"]
    mfccs = [[np.load(path) for path in mfccs_paths 
                  if genre in path and os.path.isfile(path)] for genre in genres]
    mfccs_per_genre = [len(mfccs_sublist) for mfccs_sublist in mfccs]
    labels = [genre for (genre, im_g) in zip(genres, mfccs_per_genre) for _ in range(im_g)]
    mfccs_shapes = [mfcc.shape for genre_mfccs in mfccs
                             for mfcc in genre_mfccs]
    min_h = min([t[0] for t in mfccs_shapes])
    min_w = min([t[1] for t in mfccs_shapes])
    reshaped_mfccs = [cv2.resize(mfcc, (min_w, min_h)) 
                       for genre_mfccs in mfccs
                       for mfcc in genre_mfccs]
    flattened_mfccs = np.array([mfcc.flatten() for mfcc in reshaped_mfccs])
    
    classes=np.asarray(labels)
    
    cms_test_mfccs, accs_test_mfccs, cms_train_mfccs, accs_train_mfccs = cross_validation(SVC(gamma='auto', kernel='rbf'), 
                                                                                          flattened_mfccs, 
                                                                                          classes, 
                                                                                          random_state=None) #mfccs' classification accuracy
    plot_accuracy(accs_test_mfccs, 
                  accs_train_mfccs, 
                  save_plot="C:\\Users\\Admin\\Documents\\python", 
                  plot_name="mfccs_")
    
    persistence_image_paths = [os.path.join(path_to_data_folder, f) 
                               for f in os.listdir(path_to_data_folder) 
                               if ".npy" in f and "PI" in f]
    genres = ["blues", "classical", "country", "disco", "hiphop", 
              "jazz", "metal", "pop", "reggae", "rock"]
    pers_imgs = [[np.load(path) for path in persistence_image_paths 
                  if genre in path and os.path.isfile(path)] for genre in genres]
    imgs_per_genre = [len(pers_imgs_sublist) for pers_imgs_sublist in pers_imgs]
    labels = [genre for (genre, im_g) in zip(genres, imgs_per_genre) for _ in range(im_g)]
    imgs_shapes = [img.shape for genre_imgs in pers_imgs
                             for img in genre_imgs]
    min_h = min([t[0] for t in imgs_shapes])
    min_w = min([t[1] for t in imgs_shapes])
    reshaped_images = [cv2.resize(img, (min_w, min_h)) 
                       for genre_imgs in pers_imgs
                       for img in genre_imgs]
    flattened_images = np.array([img.flatten() for img in reshaped_images])
    
    umap_proj_plot(flattened_images,genres,labels) #project PIs on a plane with UMAP's dimension reduction
        
    cms_test_PIs, accs_test_PIs, cms_train_PIs, accs_train_PIs = cross_validation(SVC(gamma='auto', kernel='rbf'), 
                                                                                  flattened_images, 
                                                                                  classes, 
                                                                                  random_state=None) #PIs classification accuracy 
                                                                             
    plot_accuracy(accs_test_PIs, 
                  accs_train_PIs, 
                  save_plot="C:\\Users\\Admin\\Documents\\python", 
                  plot_name="PIs_")
        
    
if __name__=="__main__":
    main()
