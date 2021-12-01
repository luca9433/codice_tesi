# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 23:54:50 2021

@author: Admin
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy
from scipy import ndimage 
import PIL

import librosa 
import librosa.display
import IPython.display as ipd

import persim
from ripser import ripser, lower_star_img
from persim import PersistenceImager, plot_diagrams
import gudhi
from itertools import product

import time
import numpy as np
from sklearn import datasets
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt



from pylab import  show

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap.umap_ as umap
import os
import math
import requests


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import cv2

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def create_audio_object(data):
    """
    

    Parameters
    ----------
    data : numpy array, list, unicode, str or bytes
    Can be one of

      * Numpy 1d array containing the desired waveform (mono)
      * Numpy 2d array containing waveforms for each channel.
        Shape=(NCHAN, NSAMPLES). For the standard channel order, see
        http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx
      * List of float or integer representing the waveform (mono)
      * String containing the filename
      * Bytestring containing raw PCM data or
      * URL pointing to a file on the web.

    Returns
    
    IPython.lib.display.Audio
    -------

    """
    signal, sr = librosa.load(data)
    return ipd.Audio(data,rate=sr)
    

    
def extract_mfccs(data):
    signal, sr = librosa.load(data)
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=128)
    return(mfccs)

def save_mfccs(path_to_genres_folder="C:\\Users\\Admin\\Documents\\python\\Data\\genres_original"):
    genre_paths = [os.path.join(path_to_genres_folder, f) 
                               for f in os.listdir(path_to_genres_folder)
                                if ".idea" not in f]
    for genre_path in genre_paths:
        for i in range(len(os.listdir(genre_path))):
                   np.save(os.path.splitext(os.listdir(genre_path)[i])[0],
                           extract_mfccs(os.path.join(genre_path,os.listdir(genre_path)[i])))

def cross_validation(pipeline, X, y, n_folds=5, random_state=None):
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=False) # TODO: check doc for shuffle
    cms, accs = [], []
    cms_train, accs_train = [], []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = make_pipeline(pipeline)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cms.append(confusion_matrix(y_test, y_pred)) # TODO: verify the func returns a matrix (np.ndarray)
        accs.append(accuracy_score(y_test, y_pred)) # TODO: verify that the func returns an accuracy score (float)
        y_pred_train = clf.predict(X_train)
        cms_train.append(confusion_matrix(y_train, y_pred_train))
        accs_train.append(accuracy_score(y_train, y_pred_train))
        
    return cms, accs, cms_train, accs_train


def plot_accuracy(accs_test, accs_train, ax = None, save_plot=None, plot_name=None):
    if ax is None:
        f, ax = plt.subplots()
        
    assert isinstance(accs_test, list) and isinstance(accs_train, list), "Please, provide lists as input instead of {}".format(type(accs_test))
    df = pd.DataFrame({"acc": accs_test + accs_train, "is_test": ["test" for _ in accs_test] + ["train" for _ in accs_train]})
    sns.barplot(x="is_test", y="acc", data=df, ax=ax)
    if save_plot is not None and plot_name is not None:
        f.savefig(os.path.join(save_plot, plot_name+"accuracy_barplot.svg"))
        f.savefig(os.path.join(save_plot, plot_name+"accuracy_barplot.png"))
    return df
    
    
def visualise(data):
    plt.figure(figsize=(10,5))
    librosa.display.specshow(data,x_axis="time")
    plt.colorbar(format="%+2f")
    plt.savefig('testplot.jpg')
    plt.show()

def make_life_finite(data):
    """
    

    Parameters
    ----------
    data : ndarray.

    Returns
    -------
    list.

    """
    cps = data.tolist()
    max_finite_life=np.nanmax([c[1] for c in cps if c[1]!=np.inf])
    finite_cps=[c if c[1]<=max_finite_life else [c[0],max_finite_life+1] for c in cps]
    return finite_cps


class Audio:
    def __init__(self,path):
        self.path = path
            
    def get_diagram(self):
        dgm=lower_star_img(extract_mfccs(self.path))
        return make_life_finite(dgm)  

def save_PDs(path_to_genres_folder="C:\\Users\\Admin\\Documents\\python\\Data\\genres_original"):
    genre_paths = [os.path.join(path_to_genres_folder, f) 
                               for f in os.listdir(path_to_genres_folder)
                                if ".idea" not in f]
    for genre_path in genre_paths:
        for i in range(len(os.listdir(genre_path))):
            np.save("PD_"+os.path.splitext(os.listdir(genre_path)[i])[0],
                    Audio(os.path.join(genre_path,os.listdir(genre_path)[i])).get_diagram())
            



def p_bottleneck(data_0,data_1): #Persim package's Bottleneck distance 
    """
    Calculate the Bottleneck distance between the 0-persistence diagrams
    corrisponding to data_0 and data_1 respectively 

    Parameters
    ----------
    data : ndarray
    data_1 : ndarray

    Returns
    -------
    float

    """
    distance_bottleneck = persim.bottleneck(data_0, data_1)
    return distance_bottleneck  


def g_bottleneck_approx(dgm_0,dgm_1): #Bottleneck distance con gudhi
    return gudhi.bottleneck_distance(dgm_0, dgm_1, 0.1)
    

def g_bottleneck(dgm_0,dgm_1):
    return gudhi.bottleneck_distance(dgm_0,dgm_1)
    
def save_PIs(path_to_PDs_folder='C:\\Users\\Admin\\Documents\\python'):
    PD_genre_paths = [os.path.join(path_to_PDs_folder, f)
                               for f in os.listdir(path_to_PDs_folder) 
                               if "PD" in f]
    diagrams=[]
    for PD_genre_path in PD_genre_paths:
        for i in range(len(os.listdir(PD_genre_path))):
            diagrams.append(np.load(os.path.join(PD_genre_path,os.listdir(PD_genre_path)[i])))
    
    pimgr = PersistenceImager(pixel_size=1)
    pimgr.fit(diagrams)
    
    j=0
    for PD_genre_path in PD_genre_paths:
        for i in range(len(os.listdir(PD_genre_path))):
                   np.save(os.path.splitext(os.listdir(PD_genre_path)[i])[0].replace("PD","PI"),
                           pimgr.transform(diagrams[j]))
                   j+=1
    
    
def main(path_to_data_folder="C:\\Users\\Admin\\Documents\\python",
         save_path="C:\\Users\\Admin\\Documents\\python"):
    
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
    reducer = umap.UMAP()
    reducer.fit(flattened_images)
    projector = reducer.transform(flattened_images)
    f, ax = plt.subplots(figsize=(10,10))

    for i, genre in enumerate(genres):
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
    
    
    

    



    





    

    
    
    
    

    



    





    
