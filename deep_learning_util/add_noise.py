import scipy.io as sio
import numpy as np
import os
from pathlib import Path
import wave
import normalize_wav

def open_wave(path):
    f = wave.open(path,'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    wave_data = f.readframes(nframes)
    wave_data = np.fromstring(wave_data, dtype=np.short)
    f.close()
    return wave_data, params[:3]

if os.environ['COMPUTERNAME']=='Y':

else:
    data_path = '\\big_scale\\time_stretch\\'
    save_path = '\\big_scale\\add_noise\\'

if (Path(save_path).exists()==False):
    os.mkdir(save_path)


target_word = ['zui_xiao_liang_du']

snrs = [-10,-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20] # dB
wave_time = 9000 # 4000ms
Fs = 16000
wave_len = int((wave_time/1000)*Fs)
SRC_WAVE_POW = 100000 # sum(Am)

print('start...')
print('prepare noise...')
# open the noise source
noise_list = os.listdir(data_path+'_background_noise_\\')
noise_wave = {}
for i, noise_file in enumerate(noise_list):
    noise_data, _ = open_wave(data_path+'_background_noise_\\' + noise_file)
    noise_wave[str(i)] = noise_data

print('prepare dataset...')
for target in target_word:
    if (Path(save_path+target).exists()==False):
        os.mkdir(save_path+target)
    file_list = os.listdir(data_path+target)
    print(target)
    for file in file_list:
        # source wave processing
        src_wave, params = open_wave(data_path+target+'\\'+file)
        if len(src_wave) > wave_len:
            src_wave = src_wave[:wave_len]
        elif len(src_wave) < wave_len:
            padding = np.zeros([int(wave_len-len(src_wave)),], dtype=np.int16)
            src_wave = np.concatenate([src_wave, padding])
        else:
            pass
        # clip a 4000ms noise from noise_wave, with random SNR
        noise = noise_wave[str(np.random.randint(len(noise_list)))]
        rand_start = np.random.randint(0, len(noise)-wave_len)
        noise = noise[rand_start:rand_start+wave_len]
        # merge
        snr = snrs[np.random.randint(len(snrs))]
        src_wave = normalize_wav.norm_by_pow(src_wave, SRC_WAVE_POW)
        noise_pow = np.sum(np.abs(src_wave))*10**(-snr/20)
        noise_current_pow = np.sum(np.abs(noise))
        noise = np.floor(noise*(noise_pow/noise_current_pow))
        noise = noise.astype(np.int16)
        final_wave = src_wave + noise
        # save
        if (Path(save_path+target).exists()==False):
            os.mkdir(save_path+target)
        save_file = file[:file.find('.')+3] + '_snr%d.wav'%(snr)
        with wave.open(save_path+target+'\\'+save_file,'wb') as f:
            f.setnchannels(params[0])
            f.setsampwidth(params[1])
            f.setframerate(params[2])
            f.writeframes(final_wave.tostring())
print('end...')
