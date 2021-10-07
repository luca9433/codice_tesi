# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 23:54:50 2021

@author: Admin
"""

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
    
def main():
    data_0="C:\\Users\\Admin\\Documents\\python\\Data\\genres_original\\blues\\blues.00000.wav"
    data_1="C:\\Users\\Admin\\Documents\\python\\Data\\genres_original\\blues\\blues.00001.wav"
    create_audio_object(data_0)
    create_audio_object(data_1)
    visualise(extract_mfccs(data_0))
    visualise(extract_mfccs(data_1))
    lower_star_filtration(extract_mfccs(data_0))
    lower_star_filtration(extract_mfccs(data_1))
    dgm_0=make_life_finite(lower_star_img(extract_mfccs(data_0)))
    dgm_1=make_life_finite(lower_star_img(extract_mfccs(data_1)))
    print(g_bottleneck_approx(dgm_0,dgm_1))
    print(g_bottleneck(dgm_0,dgm_1))
    
if __name__=="__main__":
    main()
    
#Test Git Bash 
    
    
    

    



    





    
