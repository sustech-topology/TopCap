#coding=utf-8
from sphfile import SPHFile
import glob
import os
import numpy as np
from scipy.io import wavfile
import shutil

import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import soundfile as sf
from gudhi.point_cloud import timedelay
from numpy import argmax
import math
from ripser import ripser
from persim import plot_diagrams



# 把TIMIT文件的Train文件夹里的所有文件放在一个文件夹下面，并且把子文件夹路径写入到文件名中
#Put all the files in the Train folder of the TIMIT dataset into one folder, and include the subfolder path in the file names.

# 源文件夹路径
source_folder = 'D:\\phonetic\\dataset'

# 目标文件夹路径
target_folder = 'D:\\phonetic\\dataset1'

# 遍历源文件夹下的所有文件夹和文件
for speaker_folder in os.listdir(source_folder):
    speaker_path = os.path.join(source_folder, speaker_folder)
    
    if os.path.isdir(speaker_path):
        for root, dirs, files in os.walk(speaker_path):
            for file in files:
                file_path = os.path.join(root, file)
                #获取子文件夹名
                nomatter_name, subfolder_name = os.path.split(root)                
                
                # 获取文件名和文件后缀
                file_name, file_extension = os.path.splitext(file)
                
                # 构造新的文件名，将文件夹名添加到文件名中
                new_file_name = f"{os.path.basename(speaker_path)}_{subfolder_name}_{file}"
                
                # 目标文件路径
                target_file_path = os.path.join(target_folder, new_file_name)
                
                # 复制文件并重命名
                shutil.copy(file_path, target_file_path)



# 原始文件播放不了是因为虽然显示的是WAV文件但实际上是SPH文件，读取后把文件拓展名替换为wav
if __name__ == "__main__":
    path = 'D:\\phonetic\\dataset1\\*.wav'
    sph_files = glob.glob(path)
    print(len(sph_files),"train utterances")
    for i in sph_files:
        sph = SPHFile(i)
        # 音频数据和采样率
        audio_data = sph.content  # 读取音频数据内容
        sample_rate = 16000
        # 指定文件夹路径和文件名， 分割文件路径
        name,ext=os.path.splitext(i)
        # 分割文件名和文件夹名
        folder_name, file_name = os.path.split(name)

        # 你的目标文件夹路径
        output_folder = "D:\\phonetic\\dataset2"  
        
        # 将数组写入为wav音频文件
        wavfile.write( output_folder+"\\"+file_name+"_nn.wav", sample_rate, audio_data.astype(np.int16))

        #os.remove(i)    # 不用删除原始SPH语音文件
    print("Completed")


M=100
max_edge_length=1
samplerate=16000
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





#将Train集中的所有选中音素切割出来，做tde+PD,记录出生死亡时间和清浊音类别到csv文件的三列中。
#Cut out all selected specific phonemes from the Train set, perform tde+PD, and record birth-death times and voiced or voiceless status into three columns in a CSV file.

# 两个音素列表
voiced_phones=['v','l','ng','m','n','y','zh']
voiceless_phones=['f','th','t','s','k','ch']

# 音频文件夹路径
audio_folder = 'D:\\phonetic\\dataset2'
# PHN文件夹路径
phn_folder = 'D:\\phonetic\\dataset1'


# 输出TopCap信息的csv文件名
csv_name = 'PD_TIMIT.csv'


# 打开文件进行写入
with open(csv_name, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)



# 目标文件夹路径
output_folder_1 = "D:\\phonetic\\TIMIT_voiced"  
output_folder_2 = "D:\\phonetic\\TIMIT_voiceless" 


# 遍历音频文件夹中的所有文件
for audio_file in os.listdir(audio_folder):
    if audio_file.endswith('.wav'):
        audio_file_path = os.path.join(audio_folder, audio_file)
        
        # 读取音频wav文件
        sig,sample_rate=sf.read(audio_folder+'\\'+audio_file)
        
        # 找到对应的PHN文件
        filename=os.path.splitext(audio_file)[0][:-3]
        phn_file = os.path.splitext(audio_file)[0][:-3] + '.phn'
        phn_file_path = os.path.join(phn_folder, phn_file)
        
        # 如果PHN文件存在
        if os.path.exists(phn_file_path):
            # 读取PHN文件内容
            with open(phn_file_path, 'r') as phn_file:
                for line in phn_file:
                    start_time, end_time, phone = line.split()
                    
                    # 检查音素是否在有声音素列表中
                    if phone in voiced_phones:
                        # 处理浊辅音音素, 可以根据起止时间切割音频
                        print(f"Found voiced phone '{phone}' from {start_time} to {end_time} in {audio_file}")
                        # 在这里可以将符合条件的音素段切割出来,存入文件夹
                        segment=sig[int(start_time): int(end_time)]
                        # 将数组写入为wav音频文件, 将浮点数数据乘以 32767（int16 的最大值），并转换为 int16 类型。
                        wavfile.write( output_folder_1+"\\"+filename+'_'+phone+'_'+start_time+".wav", sample_rate, (segment * 32767).astype(np.int16))
                        
                    elif phone in voiceless_phones:
                        # 处理清辅音音素, 可以根据起止时间切割音频
                        print(f"Found voiceless phone '{phone}' from {start_time} to {end_time} in {audio_file}")
                        # 在这里可以将符合条件的音素段切割出来,存入文件夹
                        segment=sig[int(start_time): int(end_time)]
                        # 将数组写入为wav音频文件, 将浮点数数据乘以 32767（int16 的最大值），并转换为 int16 类型。
                        wavfile.write( output_folder_2+"\\"+filename+'_'+phone+'_'+start_time+".wav", sample_rate, (segment * 32767).astype(np.int16)) 
                        
                     

# 初始化两个空列表
valid_voiced_list=[]
valid_voiceless_list=[]

# 遍历音频文件夹中的所有文件
for audio_file in os.listdir(audio_folder):
    if audio_file.endswith('.wav'):
        audio_file_path = os.path.join(audio_folder, audio_file)
        
        # 读取音频wav文件
        sig,sample_rate=sf.read(audio_folder+'\\'+audio_file)
        
        # 找到对应的PHN文件
        filename=os.path.splitext(audio_file)[0][:-3]
        phn_file = os.path.splitext(audio_file)[0][:-3] + '.phn'
        phn_file_path = os.path.join(phn_folder, phn_file)
        
        # 如果PHN文件存在
        if os.path.exists(phn_file_path):
            # 读取PHN文件内容
            with open(phn_file_path, 'r') as phn_file:
                for line in phn_file:
                    start_time, end_time, phone = line.split()
                    
                    # 检查音素是否在有声音素列表中
                    if phone in voiced_phones:
                        # 处理浊辅音音素, 可以根据起止时间切割音频
                        print(f"Found voiced phone '{phone}' from {start_time} to {end_time} in {audio_file}")
                        # 在这里可以将符合条件的音素段切割出来,存入列表valid_voiced_list尾部
                        segment=sig[int(start_time): int(end_time)]
                        valid_voiced_list.append(segment)
                        
                    elif phone in voiceless_phones:
                        # 处理清辅音音素, 可以根据起止时间切割音频
                        print(f"Found voiceless phone '{phone}' from {start_time} to {end_time} in {audio_file}")
                        # 在这里可以将符合条件的音素段切割出来,存入列表valid_voiceless_list尾部
                        segment=sig[int(start_time): int(end_time)]
                        valid_voiceless_list.append(segment)



T_voiced=[0]*len(valid_voiced_list)
for i in range(len(valid_voiced_list)):
    T_voiced[i],corr=principle_frequency_finder(np.array(valid_voiced_list[i]))
    T_voiced[i]=T_voiced[i]
    print(f"voiced_principle_frequency_find_{i}")

delay_voiced=[round(ele*6/M) for ele in T_voiced]
for element in range(len(delay_voiced)):
    if delay_voiced[element]==0:
        delay_voiced[element]=1

T_voiceless=[0]*len(valid_voiceless_list)
for i in range(len(valid_voiceless_list)):
    T_voiceless[i],corr=principle_frequency_finder(np.array(valid_voiceless_list[i]))
    T_voiceless[i]=T_voiceless[i]
    print(f"voiceless_principle_frequency_find_{i}")

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
        if data.size > 0:
            Points=point_Cloud(data)
        else:
            continue
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
        print(f"written a row {i}")

with open(csv_name,"a",newline="") as csvfile:
    writer=csv.writer(csvfile)        
    for i in range(len(valid_voiceless_list)):
        data=valid_voiceless_list[i]
        if delay_voiceless[i]*M>len(data):
            delay_voiceless[i]=int(np.floor(len(data)/M))
        if delay_voiceless[i]==0:
            delay_voiceless[i]=1
        point_Cloud=timedelay.TimeDelayEmbedding(M, delay_voiceless[i], 5)
        if i==787:
            continue
        
        try:
        # 这里是可能抛出异常的代码
            Points=point_Cloud(data)
        except Exception as e:
        # 捕获所有异常
            print(f"An error occurred: {e}")

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
        print(f"written a row {i}")            
                    
                    
                    
                    


