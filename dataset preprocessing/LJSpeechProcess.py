# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:24:16 2024

@author: 93103
"""


    
import csv
import re
import os

# 打开原始CSV文件和目标CSV文件Open the original CSV file and the target CSV file
with open('D:\\phonetic\\test_csv\\metadata.csv', 'r',encoding='utf-8') as input_file, open('correctedmetadata.csv', 'w', newline='',encoding='utf-8') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    # 遍历原始CSV文件的每一行Traverse each row of the original CSV file
    for row in reader:
        # 将一行数据合并为字符串Combine a row of data into a string
        row_string = ' '.join(row)

        #使用 split() 函数和正则表达式方法将字符串按'LJ0'分割为多个子串，并通过列表推导式筛选出以 "LJ0" 开头的子串
        #Use the split() function and a regular expression method to split the string by 'LJ0' into multiple substrings, and filter the substrings that start with 'LJ0' using a list comprehension.
        matches =  re.split(r"(?=LJ0)", row_string)
        matches = [substring for substring in matches if substring.startswith("LJ0")]


        # 写入目标CSV文件的多行Write multiple rows to the target CSV file
        for match in matches:
            writer.writerow([match])
            
            
            
           

csv_file = 'D:\\phonetic\\test_csv\\correctedmetadata.csv'  # CSV 文件的路径
output_folder = 'textgrid_files'  # 存放生成的 TextGrid 文件的文件夹

# 创建存放 TextGrid 文件的文件夹Create a folder to store the TextGrid files
os.makedirs(output_folder, exist_ok=True)

# 打开 CSV 文件 Open CSV file
with open(csv_file, 'r',encoding='utf-8') as file:
    reader = csv.reader(file)
    
    # 逐行读取 CSV 文件Read the CSV file line by line
    for row in reader:
        if len(row) > 0:
            line = row[0]  # 获取每行的字符串Obtain the string from each line.
            filename = line[:10]  # 使用前10个字符作为文件名Use the first 10 characters as the filename
            content = line[11:]  # 去除前11个字符的内容Remove the first 11 characters from the content
            content = content[len(content)//2+1:] #除以2是因为文本有两份，后面的数字写成单词形式，取后一半方便MFA识别
            # Divide by 2 because the text is duplicated. Convert the subsequent numbers into word form and take the latter half for easier MFA recognition. 
            # 创建 TextGrid 文件的路径create the path for the TextGrid file
            file_path = os.path.join(output_folder, f'{filename}.TextGrid')
            
            # 将字符串写入 TextGrid 文件 Write the string to the TextGrid file.                
            with open(file_path, 'w',encoding='utf-8') as textgrid_file:
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
