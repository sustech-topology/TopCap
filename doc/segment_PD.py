#coding=utf-8
#从请浊辅音音频到二维拓扑特征，出生时间和持续时间。
#用傅里叶变换找基频，归一化加上噪声再归一化
#为每个音分配不同的噪声，由其序号生成种子，None表示原始数据不添加噪声
#嵌入之前先修剪，并行计算缩短时间

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
from multiprocessing import Pool, cpu_count
from functools import partial



def normalize_signal(signal):
    """
    将信号归一化到 [-1, 1] 范围
    """
    max_abs_value = np.max(np.abs(signal))
    if max_abs_value == 0:
        return signal  # 如果最大绝对值为0，则返回原始信号
    return signal / max_abs_value

def add_noise(y, seed=110, noise_type='gaussian', snr_db=5):
    """
    为语音信号添加指定类型的噪声（支持随机种子控制）
    参数：
        y : 原始音频信号
        noise_type : 噪声类型 ('gaussian', 'uniform', 'impulse')
        snr_db : 信噪比（分贝）
        seed : 随机种子（控制噪声生成的可复现性）
    返回：
        noisy_y_normalized : 加噪后并且归一化的音频信号
    """
    # 在处理之前先对原始信号进行归一化
    y_normalized = normalize_signal(y)
    
    if snr_db is None:
        return y_normalized
    
    # 创建独立随机状态（避免污染全局numpy随机种子）
    rng = np.random.RandomState(seed)
    
    # 计算原始信号功率
    signal_power = np.mean(y_normalized**2)
    
    # 根据噪声类型生成噪声
    if noise_type == 'gaussian':
        noise = rng.normal(0, 1, len(y_normalized))  # 使用rng替代np.random
    elif noise_type == 'uniform':
        noise = rng.uniform(-1, 1, len(y_normalized))
    elif noise_type == 'impulse':
        noise = np.zeros_like(y_normalized)
        num_impulses = int(len(y_normalized) * 0.001)  # 调整比例以适应不同的信号长度
        indices = rng.randint(0, len(y_normalized), num_impulses)     # 随机位置
        noise[indices] = rng.uniform(-0.5, 0.5, num_impulses)  # 随机幅值
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    # 调整噪声功率到目标SNR
    noise_power = np.maximum(np.mean(noise**2), 1e-10)  # 确保噪声功率不为零
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise * scaling_factor

    # 混合信号
    noisy_y = y_normalized + scaled_noise
    
    # 对最终的加噪信号进行归一化
    noisy_y_normalized = normalize_signal(noisy_y)
    
    return noisy_y_normalized

M=100
max_edge_length=1
samplerate=16000
# wav_fraction_finder is to find the corresponding wav signal according to interval
def wav_fraction_finder(start_time, end_time,sig):
    sig_fraction=sig[int(start_time*samplerate):int(end_time*samplerate)]
    return sig_fraction

# principle_frequency_finder is to find the period of a speech signal
def principle_frequency_finder_top3(time_series):

    # 进行傅里叶变换
    n = len(time_series)  # 信号长度
    yf = np.fft.fft(time_series)  # 傅里叶变换
    xf = np.fft.fftfreq(n, d=1.0)  # 频率轴，d=1.0 表示每个样本间隔为1个单位时间

    # 计算单边幅度谱
    yf_single_side = yf[:n//2]
    magnitude_spectrum = 2.0/n * np.abs(yf_single_side)

    # 找到前三个最大峰值对应的频率（排除频率为0的情况）
    # 排除第一个值（频率为0）
    non_zero_magnitude_spectrum = magnitude_spectrum[1:]
    non_zero_xf = xf[1:]

    # 找出所有局部最大值
    peaks = []
    for i in range(1, len(non_zero_magnitude_spectrum) - 1):
        if (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i-1]) and \
        (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i+1]):
            peaks.append(i)

    if len(peaks) < 3:
        raise ValueError("Not enough local maxima to find the top three frequencies.")

    # 提取峰值对应的频率和幅度
    peak_frequencies = [non_zero_xf[p] for p in peaks]
    peak_magnitudes = [non_zero_magnitude_spectrum[p] for p in peaks]

    # 找到前三个最大峰值及其索引
    top_three_indices = sorted(range(len(peak_magnitudes)), key=lambda k: peak_magnitudes[k], reverse=True)[:3]
    frequencies_estimated = [peak_frequencies[i] for i in top_three_indices]
    periods_estimated = [1.0 / freq for freq in frequencies_estimated]

    return (periods_estimated[0])

def principle_frequency_finder(time_series):

    # 进行傅里叶变换
    n = len(time_series)  # 信号长度
    yf = np.fft.fft(time_series)  # 傅里叶变换
    xf = np.fft.fftfreq(n, d=1.0)  # 频率轴，d=1.0 表示每个样本间隔为1个单位时间

    # 计算单边幅度谱
    yf_single_side = yf[:n//2]
    magnitude_spectrum = 2.0/n * np.abs(yf_single_side)

    # 排除第一个值（频率为0）
    non_zero_magnitude_spectrum = magnitude_spectrum[1:]
    non_zero_xf = xf[1:]

    # 找出所有局部最大值
    peaks = []
    for i in range(1, len(non_zero_magnitude_spectrum) - 1):
        if (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i-1]) and \
        (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i+1]):
            peaks.append(i)

    # 提取峰值对应的频率和幅度
    peak_frequencies = [non_zero_xf[p] for p in peaks]
    peak_magnitudes = [non_zero_magnitude_spectrum[p] for p in peaks]
    
    # 检查peak_magnitudes是否为空
    if len(peak_magnitudes) == 0:
        return 1  # 或者根据需要选择其他默认返回值
    
    # 找到最大峰值的索引
    top_index = np.argmax(peak_magnitudes)
    # 提取对应的频率和周期
    frequency_estimated = peak_frequencies[top_index]
    period_estimated = 1.0 / frequency_estimated
    
    return period_estimated
    

def principle_frequency_finder_acf(sig):
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

    return max_index


                    



#持续同调计算

def process_single_item(args, data_category=1, M=100):
    """
    并行处理单个数据项的通用函数
    data_category: 1=voiced, 2=voiceless
    """
    i, data, delay_values, name_list = args
    
    try:
        # 1. 修剪音频信号的头尾部分（去除振幅 < 0.03 的部分）
        valid_indices = np.where(np.abs(data) > 0.03)[0]
        if len(valid_indices) == 0:
            return None  # 如果全部是无效信号，直接返回 None
        
        head = valid_indices[0]
        tail = valid_indices[-1]
        trimmed_data = data[head : tail + 1]
        
        # 2. 检查修剪后的信号长度是否 > 500
        if len(trimmed_data) < 500:
            return None  # 如果信号太短，丢弃
        
        # 3. 更新 data 为修剪后的信号
        data = trimmed_data

        # 延迟参数处理
        if delay_values[i] * M > len(data):
            delay_values[i] = int(np.floor(len(data) / M))
        if delay_values[i] == 0:
            delay_values[i] = 1
            
        # 时间延迟嵌入
        tau= delay_values[i]
    
        point_Cloud = timedelay.TimeDelayEmbedding(M, tau, 5)
        
        if data.size == 0:
            return None
            
        # 核心计算
        Points = point_Cloud(data)
        if len(Points) < 40 or np.isnan(Points).any():
            return None
            
        # 持续同调计算
        dgms = ripser(Points, maxdim=1)['dgms'][1]
        if dgms.size == 0:
            return None
            
        persistent_time = [ele[1]-ele[0] for ele in dgms]
        index = np.argmax(persistent_time)
        
        # 构造返回结果（保留原始索引）
        return (
            i,
            name_list[i],
            dgms[index][0],
            persistent_time[index],
            data_category
        )
        
    except Exception as e:
        print(f"Error processing item {i}: {str(e)}")
        return None

def parallel_process_to_csv(csv_name, data_list, delay_list, name_list, category, workers=None):
    """
    并行处理主函数
    category: 1=voiced, 2=voiceless
    """
    # 创建进程池
    workers = workers or max(1, cpu_count()-1)
    
    # 生成任务参数（带原始索引）
    task_args = [(i, data, delay_list, name_list) 
                for i, data in enumerate(data_list)]
    
    with Pool(workers) as pool:
        # 创建偏函数固定category参数
        processor = partial(process_single_item, data_category=category)
        
        # 并行处理（保持顺序）
        results = []
        for idx, result in enumerate(pool.imap(processor, task_args, chunksize=50)):
            if result:
                results.append(result)
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(data_list)} items")
                
    # 按原始索引排序后批量写入
    with open(csv_name, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for res in sorted(results, key=lambda x: x[0]):
            writer.writerow(res)  
            
    print(f"Category {category} completed. Total {len(results)} valid entries.")





if __name__ == "__main__":
    # 主程序初始化代码
    M = 100 #嵌入维数
    
    # 定义音频文件夹路径
    audio_folder = 'D:\\phonetic\\LibriProcess\\train500_audio_segment'  # 替换为实际路径
 

    # 输出TopCap信息的csv文件名
    csv_name = 'D:\\phonetic\\All_dataset\\Libri_train500.csv'

    # 初始化两个空列表,记录音频
    valid_voiced_list = []
    valid_voiceless_list = []

    # 初始化两个空列表,记录文件名
    name_voiced_list = []
    name_voiceless_list = []

    # 定义子文件夹
    voiced_folder = os.path.join(audio_folder, 'voiced')
    voiceless_folder = os.path.join(audio_folder, 'voiceless')

    failed_files = []  # 记录读取失败的文件

    def read_audio_file(file_path, filename):
        try:
            if filename.endswith('.wav'):
                sample_rate, data = wavfile.read(file_path)
            elif filename.endswith('.flac'):
                data, sample_rate = sf.read(file_path)
            else:
                return None, None
            return data, sample_rate
        except Exception as e:
            failed_files.append((filename, str(e)))
            return None, None

    # 读取 voiced 文件夹中的音频文件（WAV 和 FLAC）
    for filename in os.listdir(voiced_folder):
        if filename.endswith(('.wav', '.flac')):
            file_path = os.path.join(voiced_folder, filename)
            data, sample_rate = read_audio_file(file_path, filename)
            if data is not None:
                valid_voiced_list.append(data)
                name_voiced_list.append(filename)
                print(f'成功读取: {filename}')

    # 读取 voiceless 文件夹中的音频文件（WAV 和 FLAC）
    for filename in os.listdir(voiceless_folder):
        if filename.endswith(('.wav', '.flac')):
            file_path = os.path.join(voiceless_folder, filename)
            data, sample_rate = read_audio_file(file_path, filename)
            if data is not None:
                valid_voiceless_list.append(data)
                name_voiceless_list.append(filename)
                print(f'成功读取: {filename}')

    # 输出统计信息
    print(f'有效 Voiced 文件数: {len(valid_voiced_list)}')
    print(f'有效 Voiceless 文件数: {len(valid_voiceless_list)}')



    # 加噪声（每个音频独立种子）
    noisy_voiced_list = []
    noisy_voiceless_list = []

    noise_params = {
        'noise_type': 'gaussian',
        'snr_db': None,
        'base_seed': 110  # 基础种子，每个音频在此基数上增加索引
    }

    for idx, y in enumerate(valid_voiced_list):
        # 为每个音频生成唯一种子（base_seed + 索引）
        current_seed = noise_params['base_seed'] + idx
        
        noisy_y = add_noise(
            y, 
            seed=current_seed,  # 使用动态生成的独立种子
            noise_type=noise_params['noise_type'],
            snr_db=noise_params['snr_db'],
        )
        noisy_voiced_list.append(noisy_y)

    print(f"已生成 {len(noisy_voiced_list)} 个独立加噪音频")

    for idx, y in enumerate(valid_voiceless_list):
        # 为每个音频生成唯一种子（base_seed + 索引）
        current_seed = noise_params['base_seed'] + idx
        
        noisy_y = add_noise(
            y, 
            seed=current_seed,  # 使用动态生成的独立种子
            noise_type=noise_params['noise_type'],
            snr_db=noise_params['snr_db'],
        )
        noisy_voiceless_list.append(noisy_y)

    print(f"已生成 {len(noisy_voiceless_list)} 个独立加噪音频")

    valid_voiced_list = noisy_voiced_list
    valid_voiceless_list = noisy_voiceless_list


    #计算TopCap

    #计算delay量（仅在主进程执行一次）
    k=6

    T_voiced=[0]*len(valid_voiced_list)
    for i in range(len(valid_voiced_list)):
        T_voiced[i]=principle_frequency_finder(np.array(valid_voiced_list[i]))
        T_voiced[i]=T_voiced[i]
        print(f"voiced_principle_frequency_find_{i}")

    delay_voiced=[round(ele*k/M) for ele in T_voiced]
    for element in range(len(delay_voiced)):
        if delay_voiced[element]==0:
            delay_voiced[element]=1

    T_voiceless=[0]*len(valid_voiceless_list)
    for i in range(len(valid_voiceless_list)):
        T_voiceless[i]=principle_frequency_finder(np.array(valid_voiceless_list[i]))
        T_voiceless[i]=T_voiceless[i]
        print(f"voiceless_principle_frequency_find_{i}")

    delay_voiceless=[round(ele*k/M) for ele in T_voiceless]
    for element in range(len(delay_voiceless)):
        if delay_voiceless[element]==0:
            delay_voiceless[element]=1


    # 调用并行处理
    # 处理voiced数据
    parallel_process_to_csv(
        csv_name,
        valid_voiced_list,
        delay_voiced,
        name_voiced_list,
        category=1,
        workers=16
    )

    # 处理voiceless数据 
    parallel_process_to_csv(
        csv_name,
        valid_voiceless_list,
        delay_voiceless,
        name_voiceless_list,
        category=2,
        workers=16
    )
