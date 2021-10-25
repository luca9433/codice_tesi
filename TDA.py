# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 23:54:50 2021

@author: Admin
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage 
import PIL

import librosa 
import librosa.display
import IPython.display as ipd

import persim
from persim import PersistenceImager, plot_diagrams
import gudhi
from itertools import product

import time
import numpy as np
from sklearn import datasets
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt


import umap.umap_ as umap

from pylab import  show

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import h5py
import matplotlib.pyplot as plt
import umap
import os
import math
import requests


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import umap.umap_ as umap
import cv2

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

def visualise(data):
    plt.figure(figsize=(10,5))
    librosa.display.specshow(data,x_axis="time")
    plt.colorbar(format="%+2f")
    plt.savefig('testplot.jpg')
    plt.show()
    
def lower_star_filtration(img, plot=False):
    """
    construct a lowerstar filtration (sublevelset filtration) 
    on an image and calculate corresponding 0-persistence
    diagram (H_0)

    Parameters
    ----------
    data : ndarray.

    Returns
    -------
    ndarray.

    """
    dgm = lower_star_img(img)
    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(img)
        plt.colorbar()
        plt.title("Test Image")
        plt.subplot(122)
        plot_diagrams(dgm)
        plt.title("0-D Persistence Diagram")
        plt.tight_layout()
        plt.show()
    return dgm

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


def p_bottleneck(data_0,data_1): #Bottleneck distance con persim
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

def get_file_paths(dirname):
    file_paths = []  
    for root, directories, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  
    return file_paths

class Audio:
    def __init__(self,path):
        self.path = path
            
    def get_diagram(self):
        dgm=lower_star_img(extract_mfccs(self.path))
        return make_life_finite(dgm)
    
def Persistence_Image(data, plot=False):
    pimgr = PersistenceImager(pixel_size=1)
    pimgr.fit(data) 
    imgs = pimgr.transform(data)
    if plot:
        fig, axs = plt.subplots(1, 1, figsize=(10,5)) #da rivedere parte dentro l'if
        axs.set_title("Persistence Image")
        pimgr.plot_image(imgs, ax=axs)
        plt.tight_layout()
    return imgs
    
    
def main(path_to_data_folder="C:\\Users\\Admin\\Documents\\python"):
    
    persistence_image_paths = [os.path.join(path_to_data_folder, f) 
                               for f in os.listdir(path_to_data_folder) 
                               if ".npy" in f]
    genres = ["blues.00", "classical.00", "country.00", "disco.00", "hiphop.00", 
              "jazz.00", "metal.00", "pop.00", "reggae.00", "rock.00"]
    pers_imgs = [[np.load(path) for path in persistence_image_paths 
                  if genre in path] for genre in genres]
    imgs_per_genre = [len(pers_imgs_sublist) for pers_imgs_sublist in pers_imgs]
    labels = [genre for (genre, im_g) in zip(genres, imgs_per_genre) for _ in range(im_g)]
    imgs_shapes = [img.shape for genre_imgs in pers_imgs
                             for img in genre_imgs]
    min_h = min([t[0] for t in imgs_shapes])
    min_w = min([t[1] for t in imgs_shapes])
    min_shape = (min_h, min_w)
    reshaped_images = [cv2.resize(img, (min_w, min_h)) 
                       for genre_imgs in pers_imgs
                       for img in genre_imgs]
    flattened_images = np.array([img.flatten() for img in reshaped_images])
    reducer = umap.UMAP()
    reducer.fit(flattened_images.data)
    projector = reducer.transform(flattened_images.data)
    color_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = [number for (number, im_g) in zip(color_numbers, imgs_per_genre) for _ in range(im_g)]
    
    fig = plt.figure(figsize=(10, 10))
    plt.gca()
    plt.scatter(
    projector[:, 0],
    projector[:, 1],
    c=colors,
    cmap=plt.cm.nipy_spectral,
    edgecolor="k",
    label=genres
    )

if __name__=="__main__":
    main()
    
    
    

    



    





    

    
    
    
    

    



    





    
