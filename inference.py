import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
from model_bl import D_VECTOR
from collections import OrderedDict


import librosa
from synthesis import build_model
from synthesis import wavegen

from make_spect import makeSpect

import soundfile as sf

### AUDIO AND CHECKPOINT PATH
original_audio = "wavs/p225/p225_003.wav" # first audio
ref_audio = "wavs/p226/p226_003.wav" # second audio
autovc_checkpoint = 'checkpoints/autovc_1000.pt'
speaker_encoder_checkpoint = "../drive/MyDrive/MultiSpeaker_Tacotron2/3000000-BL.ckpt"
###

### GENERATE MEL
mel_org = makeSpect(original_audio)
mel_ref = makeSpect(ref_audio)
###

### GENERATE SPEAKER EMBED
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load(speaker_encoder_checkpoint)
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)

emb_org = C(torch.FloatTensor(mel_org[np.newaxis, :, :]).cuda())
emb_ref = C(torch.FloatTensor(mel_ref[np.newaxis, :, :]).cuda())
###

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = 'cuda:0'
G = Generator(32,256,512,32).eval().to(device)

g_checkpoint = torch.load(autovc_checkpoint, map_location=torch.device('cuda'))

#G.load_state_dict(g_checkpoint['model'])
G = g_checkpoint

x_org = mel_org
x_org, len_pad = pad_seq(x_org)
uttr_org = torch.FloatTensor(x_org[np.newaxis, :, :]).to(device)

with torch.no_grad():
    _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_ref)
    
if len_pad == 0:
    uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
else:
    uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

device = torch.device("cuda")
model = build_model().to(device)
checkpoint = torch.load("../drive/MyDrive/MultiSpeaker_Tacotron2/checkpoint_step001000000_ema.pth",map_location=torch.device('cuda'))
model.load_state_dict(checkpoint["state_dict"])

waveform = wavegen(model, c=uttr_trg)   
sf.write('test.wav', waveform, 16000,subtype='PCM_24')