import sys
sys.path.append('./speaker_verification/')

from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk

import torch
import librosa
import math

import numpy as np


encoder = SpeechEmbedder()
encoder.load_state_dict(torch.load("speaker_verification/final_epoch_950_batch_id_103.model"))
encoder.eval()

def concat_segs(times, segs):
    #Concatenate continuous voiced segments
    concat_seg = []
    seg_concat = segs[0]
    for i in range(0, len(times)-1):
        if times[i][1] == times[i+1][0]:
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
        else:
            concat_seg.append(seg_concat)
            seg_concat = segs[i+1]
    else:
        concat_seg.append(seg_concat)
    return concat_seg

def get_STFTs(segs):
    #Get 240ms STFT windows with 50% overlap
    sr = hp.data.sr
    STFT_frames = []
    for seg in segs:
        S = librosa.core.stft(y=seg, n_fft=hp.data.nfft,
                            win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr),pad_mode='empty')
        S = np.abs(S)**2
        mel_basis = librosa.filters.mel(sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
        for j in range(0, S.shape[1], int(.12/hp.data.hop)):
            if j + 24 < S.shape[1]:
                STFT_frames.append(S[:,j:j+24])
            else:
                break
    return STFT_frames

def align_embeddings(embeddings):
    partitions = []
    start = 0
    end = 0
    j = 1
    for i, embedding in enumerate(embeddings):
        if (i*.12)+.24 < j*.401:
            end = end + 1
        else:
            partitions.append((start,end))
            start = end
            end = end + 1
            j += 1
    else:
        partitions.append((start,end))
    avg_embeddings = np.zeros((len(partitions),256))
    for i, partition in enumerate(partitions):
        avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0)
    return avg_embeddings

def get_embedding(audio_path):
    times, segs = VAD_chunk(2, audio_path)
    #print("segs ", segs.shape)
    if segs == []:
        print('No voice activity detected')
        return None
    concat_seg = concat_segs(times, segs)
    STFT_frames = get_STFTs(concat_seg)
    STFT_frames = np.stack(STFT_frames, axis=2)
    STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0)))
    #print("STFT shape: ", STFT_frames.shape)
    embeddings = encoder(STFT_frames)
    return embeddings

def get_verification_pytorch(audio_path):
    embed1 = get_embedding(audio_path)
    #if embed1 == None: return None
    embed1 = align_embeddings(embed1.detach().numpy()) #encoder.embed_utterance(wav1)
    embed1 = np.mean(embed1, axis=0)
    return embed1