# -*- coding: utf-8 -*-
"""
Created on Thu Mar  14 16:31:12 2019

@author: Pavel Mochura
"""
import numpy as np
import mne
#from mne.viz.utils import center_cmap

chan=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4',
      'T5','T6','Fz','Cz','Pz']
# TRAIN
fnTrain=['jblhoe.vhdr', 'jblhce.vhdr', 'jbrhoe.vhdr', 'jbrhce.vhdr', 'vhlhoe.vhdr', 'vhlhce.vhdr', 'vhrhoe.vhdr', 'vhrhce.vhdr', 'dvlhoe.vhdr', 'dvlhce.vhdr', 'dvrhoe.vhdr', 'dvrhce.vhdr', 'jllhoe.vhdr', 'jllhce.vhdr', 'jlrhoe.vhdr', 'jlrhce.vhdr', 'kblhoe.vhdr', 'kblhce.vhdr', 'kbrhoe.vhdr', 'kbrhce.vhdr', 'pdlhoe.vhdr', 'pdlhce.vhdr', 'pdrhoe.vhdr', 'pdrhce.vhdr']
# VALIDATE
fnValidate=['fblhoe.vhdr', 'fblhce.vhdr', 'fbrhoe.vhdr', 'fbrhce.vhdr','jmlhoe.vhdr', 'jmlhce.vhdr', 'jmrhoe.vhdr', 'jmrhce.vhdr', 'jslhoe.vhdr', 'jslhce.vhdr', 'jsrhoe.vhdr', 'jsrhce.vhdr',  'mvlhoe.vhdr', 'mvlhce.vhdr', 'mvrhoe.vhdr', 'mvrhce.vhdr', 'nblhoe.vhdr', 'nblhce.vhdr', 'nbrhoe.vhdr', 'nbrhce.vhdr', 'rplhoe.vhdr', 'rplhce.vhdr', 'rprhoe.vhdr', 'rprhce.vhdr']

#filenames=['jllhoe.vhdr']

data_path='C:\\Users\\mochu\\Desktop\\pythonscript\\data\\'

bandpass=[]
event_id=dict(res=1,act=2,end=4,start=8)
tmin, tmax = -2.0, 0.5


def chan2idx(chan_list):
    out_list=[]
    for i in chan_list:
       out_list.append(chan.index(i))
    return out_list

def square(eeg):
    eeg = np.power(eeg, 2)
    return eeg
    
def eeg_processing(ei,fn):
    eeg=[]
    epochsERD=[]
    epochsERS=[]
    i=0
    
    if fn == 1:
        filenames=fnTrain.copy()
        output='train.txt'
    else:
        filenames=fnValidate.copy()
        output='test.txt'
        
    
    chan1=['C4']
    chan1_idx=chan2idx(chan1)
    
    chan2=['C3']
    chan2_idx=chan2idx(chan2)
    
    #zápis počtu epoch
    #f1= open("numbersOfEpochs.txt","w+") 
    
    #Načtení dat a vybrání epoch
    for file in filenames:
        if file[2] == 'l':
            chan = chan1_idx.copy()
        else:
            chan = chan2_idx.copy()
        names=[]
        eeg.append(mne.io.read_raw_brainvision("EEG\\"+file))
        eeg[i].load_data()
       
        names.append(eeg[i].ch_names) 
        
        #příprava dat pro ers
        ers = eeg[i].copy()
        ers.filter(14,22,fir_design="firwin")
        ers.apply_function(square)
        epochsERS.append(mne.Epochs(ers, mne.find_events(ers), event_id=ei, tmin=-2.0, tmax=0.5,baseline=None, preload=True, picks=chan))
        #epochsERS[i].plot()
        
        #příprava dat pro erd
        eeg[i].filter(8,12,fir_design="firwin")
        eeg[i].apply_function(square)
        epochsERD.append(mne.Epochs(eeg[i], mne.find_events(eeg[i]), event_id=ei, tmin=-2.0, tmax=0.5,baseline=None, preload=True, picks=chan))
        #epochsERD[i].plot()
        #f1.write("%s = %d\n," % (file, len(epochsERD[i].get_data())))
        i=i+1
        
    #f1.close()
    if ei == 2:
        f2= open(output,"w+")    
    else:
        f2= open(output,"a+")
        
    #Zprůměrování dat a finální výpočet ERD/ERS    
    for i in range(len(epochsERD)):
        epochsERD[i] = epochsERD[i].average()
        #epochsERD[i].plot()
        R=0
        ERD = epochsERD[i].copy()
        for j in range(len(epochsERD[i].data[0])):
            R = R + epochsERD[i].data[0][j]
        R= R / (1 + len(epochsERD[i].data[0]))  
        for j in range(len(epochsERD[i].data[0])):
            ERD.data[0][j] = (epochsERD[i].data[0][j] - R) / R * 100
            
        #ERD.plot()
    
        #df = ERD.to_data_frame()
        #dfr = df.rolling(window = 100, win_type='triang').mean()
        #dfr.plot()
        
        epochsERS[i] = epochsERS[i].average()
        #epochsERS[i].plot()
        R=0
        ERS = epochsERS[i].copy()
        for j in range(len(epochsERS[i].data[0])):
            R = R + epochsERS[i].data[0][j]
        R= R / (1 + len(epochsERS[i].data[0]))  
        for j in range(len(epochsERS[i].data[0])):
            ERS.data[0][j] = (epochsERS[i].data[0][j] - R) / R * 100
        
        #ERS.plot()
        
        #df = ERS.to_data_frame()
        #dfr = df.rolling(window = 100, win_type='triang').mean()
        #dfr.plot()
        for j in range(len(ERD.data[0])):
            f2.write("%.4f," % ERD.data[0][j])
        for j in range(len(ERS.data[0])):
            f2.write("%.4f," % ERS.data[0][j])
        if ei == 2:
            f2.write("1\n")
        else: 
            f2.write("0\n")    
    f2.close()            
        
  
eeg_processing(2,1)
eeg_processing(1,1)
eeg_processing(2,2)
eeg_processing(1,2)
