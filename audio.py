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
from pathlib import Path
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
    return signal, sr


    
def sampling_rates(path_folders="C:\\Users\\Admin\\Documents\\python\\Data\\genres_original"):
    genre_folders = [genre for genre in os.listdir(path_folders) if "idea" not in genre]
    sr=[]
    for genre in genre_folders:
        g=os.path.join(path_folders,genre)
        for i in os.listdir(g):
            sr.append(create_audio_object(os.path.join(g,str(i)))[1])#to do: verificare se posso usare create_audio_object
    return sr

def extract_mfccs(data):#extracts 128 mfccs from an audio wav file 
    signal, sr = librosa.load(data)
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=128)
    return(mfccs)

def save_mfccs(path_to_genres_folder="C:\\Users\\Admin\\Documents\\python\\Data\\genres_original",
               saving_path="C:\\Users\\Admin\\Documents\\python"):#saves mfccs extracted from audio tracks of the GTZAN dataset in the directory 
    genre_paths = [os.path.join(path_to_genres_folder, f)           #"C:\\Users\\Admin\\Documents\\python"
                               for f in os.listdir(path_to_genres_folder)
                               if ".idea" not in f]
    for genre_path in genre_paths:
        for i in range(len(os.listdir(genre_path))):
                np.save(os.path.join(saving_path,os.path.splitext(os.listdir(genre_path)[i])[0]),
                        extract_mfccs(os.path.join(genre_path,os.listdir(genre_path)[i])))