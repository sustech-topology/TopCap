# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:24:16 2024

@author: 93103
"""

import os
import shutil

def move_files(src_folder, dest_folder):
    # Ensure that the target folder exists.
    os.makedirs(dest_folder, exist_ok=True)

    # Traverse the source folder and its subfolders.
    for root, _, files in os.walk(src_folder):
        for file in files:
            # Construct the full path of the source file.
            src_file = os.path.join(root, file)
            # Construct the full path of the target file.
            dest_file = os.path.join(dest_folder, file)
            # Copy the file to the target folder (using copy2 to preserve metadata).
            shutil.copy2(src_file, dest_file)
            print(f'moved {file}')

# Copy all files from a specific folder (including files in subfolders) to another folder without preserving the original subfolder structure.
source_directory = 'D:\\phonetic\\LibriSpeech\\train-other-500'  # Source folder path.
destination_directory = 'D:\\phonetic\\LibriProcess\\train500'      # Target folder path.

move_files(source_directory, destination_directory)
          
# Specify the path of the folder to traverse (.trans.txt files path).
folder_path = 'train500'
# Folder to store the generated TextGrid files.
output_folder = 'train500Textgrid'
# Create a folder to hold the TextGrid files.
os.makedirs(output_folder, exist_ok=True)

# Traverse all files in the folder.
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        txt_file = os.path.join(folder_path, filename)
        
        # Open the txt file.
        with open(txt_file, 'r', encoding='utf-8') as file:
            print(f"Contents of {filename}:")
            # Use the replace() method to remove the specified suffix.
            result = filename.replace('.trans.txt', '')
            print(f'result={len(result)}')

            # Read the txt file line by line.
            for row in file:
                row = row.strip()  # Use strip() to remove the trailing newline character.
                if len(row) > 0:
                    line = row
                    fixlen = len(result) + 5
                    # Use the first fixlen characters as the filename.
                    filename = line[:fixlen]
                    # Remove the first fixlen+1 characters from the content.
                    content = line[fixlen + 1:]
                    
                    # Create the path for the TextGrid file.
                    file_path = os.path.join(output_folder, f'{filename}.TextGrid')
                    
                    # Write the string to the TextGrid file.
                    with open(file_path, 'w', encoding='utf-8') as textgrid_file:
                        textgrid_file.write('File type = "ooTextFile"\n')
                        textgrid_file.write('Object class = "TextGrid"\n')
                        textgrid_file.write('\n')
                        textgrid_file.write('xmin = 0\n')
                        textgrid_file.write('xmax = 40.475\n')
                        textgrid_file.write('tiers? <exists>\n')
                        textgrid_file.write('size = 1\n')
                        textgrid_file.write('item []:\n')
                        textgrid_file.write('    item [1]:\n')
                        textgrid_file.write('        class = "IntervalTier"\n')
                        textgrid_file.write('        name = "words"\n')
                        textgrid_file.write('        xmin = 0\n')
                        textgrid_file.write('        xmax = 40.475\n')
                        textgrid_file.write('        intervals: size = 1\n')
                        textgrid_file.write('        intervals [1]:\n')
                        textgrid_file.write('            xmin = 0.0\n')
                        textgrid_file.write('            xmax = 40.26\n')
                        textgrid_file.write('            text = "{}"\n'.format(content))

# Move the files from the specified folder to the output folder.
move_files(folder_path, output_folder)

# Uncomment the following line to move files from 'MinSample' to 'MinCut'
# move_files('MinSample', 'MinCut')
