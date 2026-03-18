# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:24:16 2024

@author: 93103
"""

import csv
import re
import os

# Open the original CSV file and the target CSV file.
with open("../metadata.csv", 'r', encoding='utf-8') as input_file, \
     open('correctedmetadata.csv', 'w', newline='', encoding='utf-8') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    # Traverse each row of the original CSV file.
    for row in reader:
        # Combine the data in a row into a single string.
        row_string = ' '.join(row)

        # Use the split() function and a regular expression to split the string at positions preceding 'LJ0'
        # and filter the substrings that start with "LJ0" using a list comprehension.
        matches = re.split(r"(?=LJ0)", row_string)
        matches = [substring for substring in matches if substring.startswith("LJ0")]

        # Write the resulting substrings as individual rows into the target CSV file.
        for match in matches:
            writer.writerow([match])


csv_file = "~/correctedmetadata.csv"  # Path to the CSV file.
output_folder = 'textgrid_files'  # Folder to store the generated TextGrid files.

# Create the folder to store TextGrid files; if it doesn't exist, create it.
os.makedirs(output_folder, exist_ok=True)

# Open the CSV file.
with open(csv_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)

    # Read the CSV file line by line.
    for row in reader:
        if len(row) > 0:
            line = row[0]  # Obtain the string from each line.
            filename = line[:10]  # Use the first 10 characters as the filename.
            content = line[11:]  # Remove the first 11 characters from the content.
            # Divide by 2 because the text is duplicated.
            # The subsequent numbers are written in word form; taking the latter half facilitates MFA recognition.
            content = content[len(content) // 2 + 1:]
            # Create the path for the TextGrid file.
            file_path = os.path.join(output_folder, f'{filename}.TextGrid')

            # Write the string into the TextGrid file.
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
