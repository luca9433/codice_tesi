# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 23:54:50 2021

@author: Admin
"""
<<<<<<< HEAD
import pandas as pd
||||||| 9c3d4c5

=======
import os
>>>>>>> a81c14d710e31ca40637cc7c95a03795a3408398
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage 
import PIL

import librosa 
import librosa.display
import IPython.display as ipd

from persim import plot_diagrams
from ripser import ripser, lower_star_img

<<<<<<< HEAD
from itertools import product

import time
import numpy as np
from sklearn import datasets
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

from ripser import Rips
from persim import PersistenceImager

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


||||||| 9c3d4c5
=======
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import umap.umap_ as umap

>>>>>>> a81c14d710e31ca40637cc7c95a03795a3408398
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

import persim


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

import gudhi

def g_bottleneck_approx(dgm_0,dgm_1): #Bottleneck distance con gudhi
    return gudhi.bottleneck_distance(dgm_0, dgm_1, 0.1)
    

def g_bottleneck(dgm_0,dgm_1):
    return gudhi.bottleneck_distance(dgm_0,dgm_1)

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
    

    
def PI_reduce(img): #reduction of persistence images to 2 dimensions
    reducer = umap.UMAP()
    reduction = reducer.fit_transform(img)
    return reduction
    

<<<<<<< HEAD
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
   
    X_blues_train = np.array([np.ndarray.flatten(p) for p in blues_images], dtype=np.float64)
    
    X_classical_train = np.array([np.ndarray.flatten(p) for p in classical_images])
    
    X_country_train = np.array([np.ndarray.flatten(p) for p in country_images])
    
    X_disco_train = np.array([np.ndarray.flatten(p) for p in disco_images])
    
    X_hiphop_train = np.array([np.ndarray.flatten(p) for p in hiphop_images])
    
    X_jazz_train = np.array([np.ndarray.flatten(p) for p in jazz_images])
    
    X_metal_train = np.array([np.ndarray.flatten(p) for p in metal_images])
    
    X_pop_train = np.array([np.ndarray.flatten(p) for p in pop_images])
    
    X_reggae_train = np.array([np.ndarray.flatten(p) for p in reggae_images])
    
    X_rock_train = np.array([np.ndarray.flatten(p) for p in rock_images])
    
    reducer = umap.UMAP(n_components=2, n_neighbors=5, random_state=42, transform_seed=42, verbose=False)
    
    reducer.fit(X_blues_train)
   
    blues_umap = reducer.transform(X_blues_train)
        
    reducer.fit(X_classical_train)
    
    classical_umap = reducer.transform(X_classical_train)
        
    reducer.fit(X_country_train) 
    
    country_umap = reducer.transform(X_country_train)
        
    reducer.fit(X_disco_train)
    
    disco_umap = reducer.transform(X_disco_train)
    
    reducer.fit(X_hiphop_train)
    
    hiphop_umap = reducer.transform(X_hiphop_train)
        
    reducer.fit(X_jazz_train)
    
    jazz_umap = reducer.transform(X_jazz_train)
        
    reducer.fit(X_metal_train)
    
    metal_umap = reducer.transform(X_metal_train)
        
    reducer.fit(X_pop_train)
    
    pop_umap = reducer.transform(X_pop_train)
        
    reducer.fit(X_reggae_train)
    
    reggae_umap = reducer.transform(X_reggae_train)
        
    reducer.fit(X_rock_train)
    
    rock_umap = reducer.transform(X_rock_train)
        
    genres_umap = np.vstack((blues_umap, classical_umap, country_umap, disco_umap, hiphop_umap, jazz_umap, metal_umap, pop_umap, reggae_umap, rock_umap))
    
    labels = ["blues"]*100+["classical"]*100+["country"]*100+["disco"]*100+["hiphop"]*100+["jazz"]*99+["metal"]*100+["pop"]*100+["reggae"]*100+["rock"]*100

    y_train = np.array(labels)
    
    colors=[0]*100+[1]*100+[2]*100+[3]*100+[4]*100+[5]*99+[6]*100+[7]*100+[8]*100+[9]*100

    fig = plt.figure(1, figsize=(10, 10))
    plt.clf()
    plt.scatter(
        genres_umap[:, 0],
        genres_umap[:, 1],
        c=colors,
        cmap=plt.cm.nipy_spectral,
        edgecolor="k",
        label=y_train,
    )
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.savefig("genres_PDs_2Dprojections_umap.svg")
    
||||||| 9c3d4c5
def main():
wav=[x for x in  get_file_paths('C:\\Users\\Admin\\Documents\\python\\Data') if '.wav' in x]
audios = [Audio(p) for p in wav]
diagrams = [a.get_diagram() for a in audios] #applico il metodo get_diagrams della classe Audio per estrarre i diagrammi di persistenza, che inserisco in una lista da passare come argomento di Persistence_Image
Persistence_Image(diagrams)
dz = {"blues":["blues.00000.npy", "blues.00001.npy"],"classical":["classical.00000.npy", "classical.00001.npy"],"country":["country.00000.npy", "country.00001.npy"],"disco":["disco.00000.npy", "disco.00001.npy"], "hiphop": ["hiphop.00000.npy", "hiphop.00001.npy"],"jazz":["jazz.00000.npy", "jazz.00001.npy"],"metal":["metal.00000.npy","metal.00001.npy"],"pop":["pop.00000.npy", "pop.00001.npy"],"reggae":["reggae.00000.npy", "reggae.00001.npy"],"rock":["rock.00000.npy", "rock.00001.npy"]}   
genres_data_0 = [PI_reduce(np.load(dz[i][0])) for i in dz.keys()]
genres_data_1 = [PI_reduce(np.load(dz[i][1])) for i in dz.keys()]
#UMAP 2D reduction of 2 tracks for each genre:
plt.figure(figsize=(20,10))
for i in range(len(genres_data_0)):
    plt.scatter(
    genres_data_0[i][:, 0],
    genres_data_0[i][:, 1],
    c=[sns.color_palette()[i]])
for i in range(len(genres_data_1)):
    plt.scatter(
    genres_data_1[i][:, 0],
    genres_data_1[i][:, 1],
    c=[sns.color_palette()[i]])
plt.legend(dz.keys())
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projections of persistence images', fontsize=24)
=======
def main(path_to_data_folder="data"):
    """
    data 
        - blues
            - blues.0000.npy
            - blues.0001.npy
        - country        
    """
    genres =
    genre_subfolders = [os.path.join(path_to_data_folder, f) 
                        for f in os.listdir(path_to_data_folder)]
    persistence_image_paths = [[os.path.join(genre_subfolder, f)
                                for f in os.listdir(genre_subfolder)
                                if ".npy" in f]
                               for genre_subfolder in genre_subfolders]
    """
    persistence_image_paths = [
        ["./data/blues/blues.0000.npy", "./data/blues/blues.0001.npy", ...],
        ["./data/country/country.0000.npy", "./data/country/country.0001.npy", ...],
    ]
    """
    labels = []
    persistent_images = [np.load(path) for path in persistence_image_paths]
    
    # dz = {"blues":["blues.00000.npy", "blues.00001.npy"],
#           "classical":["classical.00000.npy", "classical.00001.npy"],
#           "country":["country.00000.npy", "country.00001.npy"],
#           "disco":["disco.00000.npy", "disco.00001.npy"], 
#           "hiphop": ["hiphop.00000.npy", "hiphop.00001.npy"],
#           "jazz":["jazz.00000.npy", "jazz.00001.npy"],
#           "metal":["metal.00000.npy","metal.00001.npy"],
#           "pop":["pop.00000.npy", "pop.00001.npy"],
#           "reggae":["reggae.00000.npy", "reggae.00001.npy"],
#           "rock":["rock.00000.npy", "rock.00001.npy"]}   
    genres_data_0 = [PI_reduce(np.load(dz[i][0])) for i in dz.keys()]
    genres_data_1 = [PI_reduce(np.load(dz[i][1])) for i in dz.keys()]
    #UMAP 2D reduction of 2 tracks for each genre:
    plt.figure(figsize=(20,10))
    for i in range(len(genres_data_0)):
        plt.scatter(
        genres_data_0[i][:, 0],
        genres_data_0[i][:, 1],
        c=[sns.color_palette()[i]])
    for i in range(len(genres_data_1)):
        plt.scatter(
        genres_data_1[i][:, 0],
        genres_data_1[i][:, 1],
        c=[sns.color_palette()[i]])
    plt.legend(dz.keys())
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projections of persistence images', fontsize=24)
>>>>>>> a81c14d710e31ca40637cc7c95a03795a3408398

if __name__=="__main__":
    main()
    
    
    
    

    



    





    

    
    
    
    

    



    





    
