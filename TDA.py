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
from ripser import ripser, lower_star_img
from itertools import product

import time
import numpy as np
from sklearn import datasets
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

from ripser import Rips

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
    

def PI_umap_reduce(imgs): #reduction of persistence images to 2 dimensions
    X_train = np.array([np.ndarray.flatten(p) for p in imgs])
    reducer = umap.UMAP(n_components=2, n_neighbors=5, random_state=42, transform_seed=42, verbose=False)
    reducer.fit(X_train)
    genres_umap = reducer.transform(X_train)
    return genres_umap

def PI_proj_umap_plot(data, labels, colors):
    fig = plt.figure(1, figsize=(10, 10))
    plt.clf()
    plt.scatter(
        data[:, 0],
        data[:, 1],
        c=colors,
        cmap=plt.cm.nipy_spectral,
        edgecolor="k",
        label=labels,
    )
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.savefig("genres_PDs_2Dprojections_umap.svg")
    
    
    

def main(path_to_data_folder="C:\\Users\\Admin\\Documents\\python"):
    
    persistence_image_paths = [os.path.join(path_to_data_folder, f) for f in os.listdir(path_to_data_folder) if ".npy" in f]
   
    blues_images = np.array([np.load(path) for path in persistence_image_paths if "\\blues.00" in path])
    
    classical_images = np.array([np.load(path) for path in persistence_image_paths if "\\classical.00" in path])
    
    country_images = np.array([np.load(path) for path in persistence_image_paths if "\\country.00" in path])
    
    disco_images = np.array([np.load(path) for path in persistence_image_paths if "\\disco.00" in path])
    
    hiphop_images = np.array([np.load(path) for path in persistence_image_paths if "\\hiphop.00" in path])
    
    jazz_images = np.array([np.load(path) for path in persistence_image_paths if "\\jazz.00" in path])
        
    metal_images = np.array([np.load(path) for path in persistence_image_paths if "\\metal.00" in path])
    
    pop_images = np.array([np.load(path) for path in persistence_image_paths if "\\pop.00" in path])
        
    reggae_images = np.array([np.load(path) for path in persistence_image_paths if "\\reggae.00" in path])
        
    rock_images = np.array([np.load(path) for path in persistence_image_paths if "\\rock.00" in path])
   
    genres_umap = np.vstack((PI_umap_reduce(blues_images), 
                             PI_umap_reduce(classical_images), 
                             PI_umap_reduce(country_images),
                             PI_umap_reduce(disco_images), 
                             PI_umap_reduce(hiphop_images),
                             PI_umap_reduce(jazz_images), 
                             PI_umap_reduce(metal_images),
                             PI_umap_reduce(pop_images),
                             PI_umap_reduce(reggae_images),
                             PI_umap_reduce(rock_images)
                             ))
    
    
    labels = ["blues"]*100+["classical"]*100+["country"]*100+["disco"]*100+["hiphop"]*100+["jazz"]*99+["metal"]*100+["pop"]*100+["reggae"]*100+["rock"]*100
    y_train = np.array(labels)
    
    PI_proj_umap_plot(genres_umap,
                      y_train,
                      colors=[0]*100+[1]*100+[2]*100+[3]*100+[4]*100+[5]*99+[6]*100+[7]*100+[8]*100+[9]*100
                      )


if __name__=="__main__":
    main()
    
    
    
    

    



    





    

    
    
    
    

    



    





    
