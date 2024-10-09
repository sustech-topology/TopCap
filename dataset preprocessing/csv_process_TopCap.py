from praatio import textgrid
import os 
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import soundfile as sf
from gudhi.point_cloud import timedelay
import numpy as np
from numpy import argmax
import math
from ripser import ripser
from persim import plot_diagrams


#voiced_phones=['v','l','ŋ','m','n','j','ʒ']
#voiceless_phones=['f','θ','t','s','k','tʃ']
voiced_phones=['V','L','NG','M','N','Y','ZH']
voiceless_phones=['F','TH','T','S','K','CH']

M=100
max_edge_length=1
samplerate=16000
#inputPath="D:\\phonetic\\new_consonant_recognition\\speech_file_input_LJSpeech"
inputPath="D:\\phonetic\\LibriProcess\\MInCut"
#outputPath="D:\\phonetic\\new_consonant_recognition\\speech_file_output"
csv_name="PersistentDiagRefined_MinLibri.csv"



# wav_fraction_finder is to find the corresponding wav signal according to interval
def wav_fraction_finder(start_time, end_time,sig):
    sig_fraction=sig[int(start_time*samplerate):int(end_time*samplerate)]
    return sig_fraction

# head_tail_scissor is to erase signal in head and tail that has amplitude smaller than 0.05
# can also use it to see if the length of renewing signal is greater than 500 or not 
def head_tail_scissor(sig):
    valid_interval=[index for index in range(len(sig)) if (sig[index]>0.03).any()]
    if len(valid_interval)==0:
        return False,sig
    head=min(valid_interval)
    tail=max(valid_interval)
    sig=sig[head:tail+1]
    if tail-head<500:
        return False,sig
    return True,sig

# principle_frequency_finder is to find the period of a speech signal
def principle_frequency_finder(sig):
    t=int(len(sig)/2)
    corr=np.zeros(t)

    for index in np.arange(t):
        ACF_delay=sig[index:]
        L=(t-index)/2
        m = np.sum(sig[int(t-L):int(t+L+1)]**2) + np.sum(ACF_delay[int(t-L):int(t+L+1)]**2)
        r = np.sum(sig[int(t-L):int(t+L+1)]*ACF_delay[int(t-L):int(t+L+1)])
        corr[index] = 2*r/m

    zc = np.zeros(corr.size-1)
    zc[(corr[0:-1] < 0)*(corr[1::] > 0)] = 1
    zc[(corr[0:-1] > 0)*(corr[1::] < 0)] = -1

    admiss = np.zeros(corr.size)
    admiss[0:-1] = zc
    for i in range(1, corr.size):
        if admiss[i] == 0:
            admiss[i] = admiss[i-1]

    maxes = np.zeros(corr.size)
    maxes[1:-1] = (np.sign(corr[1:-1] - corr[0:-2])==1)*(np.sign(corr[1:-1] - corr[2::])==1)
    maxidx = np.arange(corr.size)
    maxidx = maxidx[maxes == 1]
    max_index = 0
    if len(corr[maxidx]) > 0:
        max_index = maxidx[np.argmax(corr[maxidx])]

    return (max_index, corr)


for fn in os.listdir(inputPath):
    fileName,ext=os.path.splitext(fn)
    if ext==".TextGrid":
        
        tg=textgrid.openTextgrid(inputPath+'\\'+fn,includeEmptyIntervals=False)
        
        # 记录音素的Tier可能有两种名字'Speaker - phone'或者'utt - phones'
        # 尝试读取 'Speaker - phone' 这个 tier
        '''try:
            phoneTier = tg.getTier('Speaker - phone')
        except KeyError:
            # 如果 'Speaker - phone' tier 不存在，则尝试读取 'utt - phones' 这个 tier
            try:
                phoneTier = tg.getTier('utt - phones')
            except KeyError:
                print("Neither 'Speaker - phone' nor 'utt - phones' tier found in the TextGrid file.")'''
        
        phoneTier = tg.getTier('phones')
        
        wavFile=inputPath+'\\'+fileName+".wav"
        sig,samplerate=sf.read(wavFile)
        voiced_list=[ele for ele in phoneTier.entries if ele[2] in voiced_phones]
        voiceless_list=[ele for ele in phoneTier.entries if ele[2] in voiceless_phones]
    
        valid_voiced_list=[head_tail_scissor(wav_fraction_finder(ele[0],ele[1],sig))[1] for ele in voiced_list if head_tail_scissor(wav_fraction_finder(ele[0],ele[1],sig))[0]]
        valid_voiceless_list=[head_tail_scissor(wav_fraction_finder(ele[0],ele[1],sig))[1] for ele in voiceless_list if head_tail_scissor(wav_fraction_finder(ele[0],ele[1],sig))[0]]

        T_voiced=[0]*len(valid_voiced_list)
        for i in range(len(valid_voiced_list)):
            T_voiced[i],corr=principle_frequency_finder(np.array(valid_voiced_list[i]))
            T_voiced[i]=T_voiced[i]

        delay_voiced=[round(ele*6/M) for ele in T_voiced]
        for element in range(len(delay_voiced)):
            if delay_voiced[element]==0:
                delay_voiced[element]=1

        T_voiceless=[0]*len(valid_voiceless_list)
        for i in range(len(valid_voiceless_list)):
            T_voiceless[i],corr=principle_frequency_finder(np.array(valid_voiceless_list[i]))
            T_voiceless[i]=T_voiceless[i]

        delay_voiceless=[round(ele*6/M) for ele in T_voiceless]
        for element in range(len(delay_voiceless)):
            if delay_voiceless[element]==0:
                delay_voiceless[element]=1

        with open(csv_name,"a",newline="") as csvfile:
            writer=csv.writer(csvfile)        
            for i in range(len(valid_voiced_list)):
                data=valid_voiced_list[i]
                if delay_voiced[i]*M>len(data):
                    delay_voiced[i]=int(np.floor(len(data)/M))
                if delay_voiced[i]==0:
                    delay_voiced[i]=1
                point_Cloud=timedelay.TimeDelayEmbedding(M, delay_voiced[i], 5)
                Points=point_Cloud(data)
                if len(Points)<40:               
                    continue
                dgms = ripser(Points,maxdim=1)['dgms']
                dgms=dgms[1]
                if dgms.size==0:
                    continue
                persistent_time=[ele[1]-ele[0] for ele in dgms]            
                index=argmax(persistent_time)
                birth_date=dgms[index][0]
                lifetime=persistent_time[index]
                writer.writerow((birth_date,lifetime,1))

        with open(csv_name,"a",newline="") as csvfile:
            writer=csv.writer(csvfile)        
            for i in range(len(valid_voiceless_list)):
                data=valid_voiceless_list[i]
                if delay_voiceless[i]*M>len(data):
                    delay_voiceless[i]=int(np.floor(len(data)/M))
                if delay_voiceless[i]==0:
                    delay_voiceless[i]=1
                point_Cloud=timedelay.TimeDelayEmbedding(M, delay_voiceless[i], 5)
                Points=point_Cloud(data)
                if len(Points)<40:               
                    continue
                dgms = ripser(Points,maxdim=1)['dgms']
                dgms=dgms[1]
                if dgms.size==0:
                    continue
                persistent_time=[ele[1]-ele[0] for ele in dgms]            
                index=argmax(persistent_time)
                birth_date=dgms[index][0]
                lifetime=persistent_time[index]
                writer.writerow((birth_date,lifetime,2))

        continue
