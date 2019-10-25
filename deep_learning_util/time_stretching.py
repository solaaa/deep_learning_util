import os
from pathlib import Path
import numpy as np
import time
target_word_en = ['zui_da_liang_du', 'zui_xiao_liang_du']

MAX_LOOP = 2 # expand each .wav MAX_LOOP times 
MIN_STRETCH = 0.6
MAX_STRETCH = 1.1

src_path = '\\big_scale\\downsample\\'
dst_path = '\\big_scale\\time_stretch\\'
if (Path(src_path).exists()==False):
    os.mkdir(src_path)
target_list = os.listdir(src_path)
if (Path(dst_path).exists()==False):
    os.mkdir(dst_path)
rubberband_path = 'E:\\rubberband\\command_line\\rubberband-1.8.2-gpl-executable-windows\\'

for target_en in target_word_en:
    target_scr_path = src_path + target_en + '\\'
    target_dst_path = dst_path + target_en + '\\'
    if (Path(target_scr_path).exists()==False):
        os.mkdir(target_scr_path)
    if (Path(target_dst_path).exists()==False):
        os.mkdir(target_dst_path)
    file_list = os.listdir(target_scr_path)
    #count = 1
    for i in range(MAX_LOOP):
        for wave_file in file_list:
            rand_time_stretch = np.random.randint(MIN_STRETCH*100, MAX_STRETCH*100)/100.0
            new_wave_file = wave_file[:wave_file.find('_')+3] + 'ratio_%.2f.wav'%(rand_time_stretch)
            cmd = rubberband_path + 'rubberband -c 6 -q --realtime -t ' + str(rand_time_stretch) + ' ' + target_scr_path + wave_file + ' ' + target_dst_path + new_wave_file
            res = os.popen(cmd)
            time.sleep(0.04) # wait
            #count += 1
        #count = 1

