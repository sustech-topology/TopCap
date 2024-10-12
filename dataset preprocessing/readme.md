# Preprocessing of datasets
- ALLSSTAR
  Use [DeleteTiersInSpeechBox](https://github.com/sustech-topology/TopCap/blob/main/dataset%20preprocessing/DeleteTiersInSpeechBox.py) to delete all other tiers in ALLSSTAR excepts "words", then use MFA to get "phones" tier.
- TIMIT
  Use [TIMIT_study](https://github.com/sustech-topology/TopCap/blob/main/dataset%20preprocessing/TIMIT_study.py) to process TIMIT dataset. 
- LJSpeech
  Use [LJSpeechProcess](https://github.com/sustech-topology/TopCap/blob/main/dataset%20preprocessing/LJSpeechProcess.py) to process LJSpeech dataset. 
  Reading the CSV: The script opens the original CSV file and reads its contents. Processing Sentences: It splits each sentence based on the identifier 
  LJ0 and filters the relevant entries. Creating TextGrid Files: For each processed sentence, a corresponding TextGrid file is generated with the 
  appropriate format for MFA.
- LibriSpeech
  Use [LibriSpeechProcess](https://github.com/sustech-topology/TopCap/blob/main/dataset%20preprocessing/LibriSpeechProcess.py) to process LibriSpeech 
  dataset. The file begins by traversing the source folder and its subfolders, gathering all files into a new directory. Next, it reads the .txt files: 
  the script opens the original txt file and reads its contents. During the processing of sentences, it splits each sentence based on several beginning 
  string characters. Finally, for each processed sentence, a corresponding TextGrid file is generated with the appropriate format for MFA.
