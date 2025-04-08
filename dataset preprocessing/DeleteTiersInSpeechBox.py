# The original files have a TextGrid. Delete the utt (sentences) and phones tiers.
# Modify the TextGrid file then send it to MFA and convert the sampling rate to 16 kHz.
import os
import shutil
from praatio import textgrid
from praatio import audio
import os 
from os.path import join
import librosa
from scipy.io.wavfile import write

# Source folder path for the original ALLSSTAR dataset.
inputPath = "../HT2"
# Source folder path.
source_folder = inputPath
# Target folder path.
destination_folder = "../Deleted_HT2"

# Ensure that the target folder exists; if it does not, create it.
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Traverse the files in the source folder and copy them to the target folder.
for filename in os.listdir(source_folder):
    source_file = os.path.join(source_folder, filename)
    destination_file = os.path.join(destination_folder, filename)
    # Copy file.
    shutil.copy(source_file, destination_file)

print("File copying complete!")

for fn in os.listdir(destination_folder):
    # Read the filename and extension separately.
    name, ext = os.path.splitext(fn)
    if ext != ".wav":
        inputFilename = os.path.join(destination_folder, name + ".TextGrid")
        tg = textgrid.openTextgrid(inputFilename, includeEmptyIntervals=False)
        # Modify the original TextGrid file to include only the words tier.
        if 'utt' in tg._tierDict:
            tg.removeTier('utt')
        if 'Speaker - phone' in tg._tierDict:
            tg.removeTier('Speaker - phone')
        if 'Speaker - word' in tg._tierDict:
            tg.renameTier('Speaker - word', 'words')
        if 'utt - phones' in tg._tierDict:
            tg.removeTier('utt - phones')
        if 'utt - words' in tg._tierDict:
            tg.renameTier('utt - words', 'words')
        tg.save(inputFilename, 'long_textgrid', False)
        continue
    # Load the .wav file with librosa and change the sampling rate from 22.05kHz (default) to 16kHz.
    y, s = librosa.load(os.path.join(destination_folder, name + ".wav"), sr=16000)
    write(os.path.join(destination_folder, name + ".wav"), 16000, y)



