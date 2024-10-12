# Preprocessing of datasets
- ALLSSTAR
  
  Use [DeleteTiersInSpeechBox](https://github.com/sustech-topology/TopCap/blob/main/dataset%20preprocessing/DeleteTiersInSpeechBox.py) to remove all 
  tiers from the original TextGrid file except for the "words" tier. Next, utilize the Montreal Forced Aligner (MFA) to analyze the modified TextGrid 
  file, which will generate a new "phones" tier. 
  
- TIMIT
  
  Use [TIMIT_study](https://github.com/sustech-topology/TopCap/blob/main/dataset%20preprocessing/TIMIT_study.py) to process TIMIT dataset. The script 
  organizes all files from the Train folder of the TIMIT dataset into a single directory, ensuring that the paths of subfolders are included in the 
  filenames to avoid any confusion with duplicate names. The original files cannot be played because they are displayed as WAV files but are actually 
  SPH files. After reading these files, the script replaces their extensions with .wav and extracts all selected phonemes from the Train set.
  
- LJSpeech
  
  Use [LJSpeechProcess](https://github.com/sustech-topology/TopCap/blob/main/dataset%20preprocessing/LJSpeechProcess.py) to process LJSpeech dataset. 
  Reading the CSV: The script opens the original CSV file and reads its contents. Processing Sentences: It splits each sentence based on the identifier 
  LJ0 and filters the relevant entries. Creating TextGrid Files: For each processed sentence, a corresponding TextGrid file is generated with the 
  appropriate format for MFA.
  
- LibriSpeech
  
  Use [LibriSpeechProcess](https://github.com/sustech-topology/TopCap/blob/main/dataset%20preprocessing/LibriSpeechProcess.py) to process LibriSpeech 
  dataset. The file begins by traversing the source folder and its subfolders, gathering all files into a new directory. Next, the script opens the 
  original txt file and reads its contents to process the sentences, it splits each sentence based on several beginning string characters. Finally, 
  for each processed sentence, a corresponding TextGrid file is generated with the appropriate format for MFA.

# As input if main experiment and comparison experiment
In our study, we consider two types of input: the main experiment and the comparison experiment. 

The main experiment employs the TopCap method [csv_process_TopCap](https://github.com/sustech-topology/TopCap/blob/main/dataset%20preprocessing/csv_process_TopCap.py), utilizing the "phones" tier in the TextGrid file to extract the corresponding phonemes, then to do time delay embedding and persistent homology to obtain topological features, which are then recorded in a CSV file. All datasets follow this streamlined process except for TIMIT, which uses its original phonemes cutting information. The relevant code for this part is included in the TIMIT_study file. 

The comparison experiment utilizes the [cut_wav](https://github.com/sustech-topology/TopCap/blob/main/dataset%20preprocessing/cut_wav.py) files to extract audio segments corresponding to the selected factors. These segments are saved in two subfolders within a single directory, serving as input for the neural network classification.
