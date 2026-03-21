# Data preprocessing

## ALLSSTAR
  
[`ALLSSTAR.py`](ALLSSTAR.py) removes all tiers from the original `.TextGrid` file except for the "words" tier.  It then uses Montreal Forced Aligner (MFA, cf. the [primary](/TopCap/primary) experiments) to generate a new "phones" tier.  

## LJSpeech

[`LJSpeech.py`](LJSpeech.py) processes the LJSpeech dataset as follows: 

- Reading the `.csv` file: The script opens the original `.csv` file and reads its contents.
- Processing sentences: It splits each sentence based on the identifier LJ0 and filters the relevant entries.
- Creating `.TextGrid` files: For each processed sentence, a corresponding `.TextGrid` file is generated with the appropriate format for MFA.  

## TIMIT

[`TIMIT.py`](TIMIT.py) organises all files from the Train folder of the TIMIT dataset into a single directory, ensuring that the paths of subfolders are included in the filenames to avoid any confusion with duplicate names.  The original files cannot be played because they are displayed as `.wav` files but are actually `.sph` files.  After reading these files, the script replaces their extensions with `.wav` and extracts all selected phones from the Train set.  

## LibriSpeech
  
[`LibriSpeech.py`](LibriSpeech.py) begins with traversing the source folder and its subfolders, gathering all files into a new directory.  Next, the script opens the original `.txt` file and reads its contents to process the sentences.  It splits each sentence based on several beginning string characters.  Finally, for each processed sentence, a corresponding `.TextGrid` file is generated with the appropriate format for MFA.  

## Inputs for TopCap and its comparison models

For TopCap, [`feature.py`](/TopCap/primary/feature.py) uses the "phones" tier in the `.TextGrid` file to extract the corresponding phones.  It then performs time-delay embedding and persistent homology to obtain topological features, which are recorded in a `.csv` file.  All datasets are fed to this streamlined process except TIMIT, which provides its original phone cutting information (see [`TIMIT.py`](TIMIT.py) above).  

The model comparison experiment uses [`cut_wav.py`](cut_wav.py) to extract speech segments corresponding to the selected factors.  These segments are saved in two subfolders within a single directory, serving as input for the neural network classifications.  
