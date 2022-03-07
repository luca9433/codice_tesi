# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 23:54:50 2021

@author: Admin
"""

import os
import numpy as np
import librosa 
import librosa.display

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

#The following function return sempling rates for audio track of the GTZAN dataset
    
def sampling_rates(path_folders="C:\\Users\\Admin\\Documents\\python\\Data\\genres_original"):
    genre_folders = [genre for genre in os.listdir(path_folders) if "idea" not in genre]
    sr=[]
    for genre in genre_folders:
        g=os.path.join(path_folders,genre)
        for i in os.listdir(g):
            sr.append(create_audio_object(os.path.join(g,str(i)))[1])
    return sr

#The following fucntion can extract mfccs from an input audio file.

def extract_mfccs(data):#extracts 128 mfccs from an audio wav file 
    signal, sr = librosa.load(data)
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=128)
    return(mfccs)

#The following function has been used to save mfccs extracted from audio file in GTZAN.

def save_mfccs(path_to_genres_folder="C:\\Users\\Admin\\Documents\\python\\Data\\genres_original",
               saving_path="C:\\Users\\Admin\\Documents\\python"):#saves mfccs extracted from audio tracks of the GTZAN dataset in the directory 
    genre_paths = [os.path.join(path_to_genres_folder, f)           #"C:\\Users\\Admin\\Documents\\python"
                               for f in os.listdir(path_to_genres_folder)
                               if ".idea" not in f]
    for genre_path in genre_paths:
        for i in range(len(os.listdir(genre_path))):
                np.save(os.path.join(saving_path,os.path.splitext(os.listdir(genre_path)[i])[0]),
                        extract_mfccs(os.path.join(genre_path,os.listdir(genre_path)[i])))