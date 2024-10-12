# The original files have a TextGrid, delete the utt(sentences) and phones tiers 
# Modify TextGrid file then sent to mfa, convert sampling rate to 16 kHz
import os
import shutil
from praatio import textgrid
from praatio import audio
import os 
from os.path import join
import librosa
from scipy.io.wavfile import write

# 源文件夹路径,原始ALLSSTAR数据集, Source folder path
inputPath="D:\\phonetic\\ALLSSTAR_reMFA\\HT2"
# 源文件夹路径Source folder path
source_folder = inputPath
# 目标文件夹路径Target folder path 
destination_folder = "D:\\phonetic\\ALLSSTAR_reMFA\\Deleted_HT2"

# 确保目标文件夹存在，如果不存在则创建Ensure that the target folder exists; if it does not, create it.
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历源文件夹中的文件,先复制到Deleted_DHR. 
# Traverse the files in the source folder and copy them to Deleted_DHR
for filename in os.listdir(source_folder):
 #   if filename.endswith(".wav"):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        # 复制文件 copy file
        shutil.copy(source_file, destination_file)

print("文件复制完成！")



for fn in os.listdir(destination_folder):
    #分别读取文件名和后缀 Read the filename and the file extension separately.
    name,ext=os.path.splitext(fn)
    if ext!=".wav":
        inputFilename=destination_folder+'\\'+name+".TextGrid"
        tg=textgrid.openTextgrid(inputFilename,includeEmptyIntervals=False)
        # modify original textgrid file to words tier only
        if 'utt' in tg._tierDict:
            tg.removeTier('utt')
        if 'Speaker - phone' in tg._tierDict:
            tg.removeTier('Speaker - phone')
        if 'Speaker - word' in tg._tierDict:
            tg.renameTier('Speaker - word','words')
        if 'utt - phones' in tg._tierDict:
            tg.removeTier('utt - phones')
        if 'utt - words' in tg._tierDict:
            tg.renameTier('utt - words','words')
        tg.save(inputFilename,'long_textgrid',False)
        continue
    y, s = librosa.load(destination_folder+'\\'+name+".wav", sr=16000)# change sample rate from 22.05kHz to 16kHz
    write(destination_folder+'\\'+name+".wav", 16000, y)



