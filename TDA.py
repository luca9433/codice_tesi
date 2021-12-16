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
        plt.imshow(img)
        plt.colorbar()
        plt.title("Test Image")
        plt.subplot(122)
        plot_diagrams(dgm)
        plt.savefig("0-PD.png")
        plt.title("0-Persistence Diagram")
        plt.tight_layout()
        plt.show()
    return dgm

def make_life_finite(data):#can be used to replace cornerlines or cornerpoints in a persistence diagram with too high death-coordinates
                           #with other cornerpoints with the same birth_coordinates and death-coordinates equal to maximum finite death_coordinate,
                           #which is not a nan, plus 1, of cornerpoints in the diagram 
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

def save_PDs(path_to_genres_folder="C:\\Users\\Admin\\Documents\\python\\Data\\genres_original",
             saving_path="C:\\Users\\Admin\\Documents\\python"): #saves PDs corresponding to the audio tracks from the GTZAN dataset in subdirectories of the "python" 
    genre_paths = [os.path.join(path_to_genres_folder, f)           #directory, one for each genre
                               for f in os.listdir(path_to_genres_folder)
                                if ".idea" not in f]
    for genre_path in genre_paths:
        os.mkdir(os.path.join(saving_path,"PD_"+Path(genre_path).parts[-1]))
        p=os.path.join(saving_path,"PD_"+Path(genre_path).parts[-1])
        for i in range(len(os.listdir(genre_path))):
            np.save(os.path.join(p,os.path.splitext(os.listdir(genre_path)[i])[0]),
                    Audio(os.path.join(genre_path,os.listdir(genre_path)[i])).get_diagram())
            

    
def save_PIs(path_to_PDs_folder='C:\\Users\\Admin\\Documents\\python'):#saves PIs corresponding to audio tracks in GTZAN dataset in the "python" directory
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
                   
def p_bottleneck(data_0,data_1): #persim package's Bottleneck distance 
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



def g_bottleneck(dgm_0,dgm_1,e=None): #gudhi package's Bottleneck distance: 
    return gudhi.bottleneck_distance(dgm_0,dgm_1,e=e)
    
    
    
    
    

    


    





    

    
    
    
    

    



    





    
