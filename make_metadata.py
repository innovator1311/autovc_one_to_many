"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
#import speaker_dct from speaker_dct


#################### 
#from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from numpy import dot
from numpy.linalg import norm
import random

import torch
import librosa
import math

from get_verification import get_verification_pytorch

num_uttrs = 10
len_crop = 128

# Directory containing mel-spectrograms
rootDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []
for speaker in sorted(subdirList):
    
    print('Processing speaker: %s' % speaker)
    
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # make speaker embedding

    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []

    for i in range(num_uttrs):
        
        #Verification Pytorch
        try:
            audio_path = "../vivos_only_wavs/{}/{}.wav".format(speaker,fileList[idx_uttrs[i]][:-4])
            emb = get_verification_pytorch(audio_path)

            if np.isnan(np.sum(emb)): 
                print("Have nan")
                continue

            embs.append(emb)
        except:
            continue
    
    assert len(embs) != 0

    utterances.append(np.mean(embs, axis=0))

    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train_speaker_embed.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)
