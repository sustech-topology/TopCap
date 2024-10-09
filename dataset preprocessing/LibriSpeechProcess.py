# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:24:16 2024

@author: 93103
"""


    


import os
import shutil

def move_files(src_folder, dest_folder):
    # 确保目标文件夹存在
    os.makedirs(dest_folder, exist_ok=True)

    # 遍历源文件夹及其子文件夹
    for root, _, files in os.walk(src_folder):
        for file in files:
            # 构建源文件的完整路径
            src_file = os.path.join(root, file)
            # 构建目标文件的完整路径
            dest_file = os.path.join(dest_folder, file)
            # 将文件复制到目标文件夹
            shutil.copy2(src_file, dest_file)  # 使用 copy2 保留文件元数据
            print(f'moved {file}')


# 将某个文件夹中的所有文件（包括子文件夹内的文件）复制到另一个文件夹中，并且不保留原来的子文件夹结构
source_directory = 'D:\\phonetic\\LibriSpeech\\train-other-500'  # 源文件夹路径

destination_directory = 'D:\\phonetic\\LibriProcess\\train500'  # 目标文件夹路径

move_files(source_directory, destination_directory)
          


# 指定要遍历的文件夹路径 # .trans.txt的路径
folder_path = 'train500' 
# 存放生成的 TextGrid 文件的文件夹
output_folder = 'train500Textgrid'  
# 创建存放 TextGrid 文件的文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        txt_file = os.path.join(folder_path, filename)
        
        # 打开 txt 文件
        with open(txt_file, 'r',encoding='utf-8') as file:
            
            print(f"Contents of {filename}:")
            # 使用 replace() 方法去除指定后缀
            result = filename.replace('.trans.txt', '')

            print(f'result={len(result)}')

            # 逐行读取 txt 文件
            for row in file:
                row=row.strip() # 使用 strip() 去除行末的换行符
                #print(row)
                if len(row) > 0:
                    line = row
                    fixlen=len(result)+5
                    filename = line[:fixlen]  # 使用前xx个字符作为文件名
                    content = line[fixlen+1:]  # 去除前xx+1个字符的内容
                    
                    # 创建 TextGrid 文件的路径
                    file_path = os.path.join(output_folder, f'{filename}.TextGrid')
                    
                    # 将字符串写入 TextGrid 文件                
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




move_files(folder_path, output_folder)


#move_files('MinSample', 'MinCut')