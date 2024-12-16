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

def visualise_mfccs(data):
    plt.figure(figsize=(10,5))
    librosa.display.specshow(extract_mfccs(data),x_axis="time")
    plt.colorbar(format="%+2f")
    plt.savefig('testplot.jpg')
    plt.show()
    
def lower_star_filtration(data):
    """
    construct a lowerstar filtration (sublevelset filtration) 
    on the mfccs image and calculate corresponding 0-persistence
    diagram (H_0)

    Parameters
    ----------
    data : ndarray.

    Returns
    -------
    ndarray.

    """
    dgm = lower_star_img(extract_mfccs(data))
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(extract_mfccs(data))
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
    cps = lower_star_img(extract_mfccs(data)).tolist()
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
    distance_bottleneck = persim.bottleneck(make_life_finite(data_0), make_life_finite(data_1))
    return distance_bottleneck  

import gudhi

def g_bottleneck_approx(data_0,data_1): #Bottleneck distance con gudhi
    message = "Bottleneck distance approximation = " + '%.2f' % gudhi.bottleneck_distance(lower_star_img(extract_mfccs(data_0)),lower_star_img(extract_mfccs(data_1)), 0.1)
    print(message)

def g_bottleneck(data_0,data_1):
    message = "Bottleneck distance value = " + '%.2f' % gudhi.bottleneck_distance(lower_star_img(extract_mfccs(data_0)),lower_star_img(extract_mfccs(data_1)))
    print(message)
    

    



    





    
