import numpy as np
import matplotlib.pyplot as p
import wave
import os

def norm_by_am(wave_data, max_volume):
    wave_data = np.floor((wave_data/max(np.abs(wave_data)))*max_volume)
    wave_data = wave_data.astype(np.short)
    return wave_data

def norm_by_pow(wave_data, target_power=100000):
    crt_power = np.sum(np.abs(wave_data))
    wave_data = np.floor((target_power/crt_power)*wave_data)
    wave_data = wave_data.astype(np.short)
    return wave_data

def norm_wave(src_path, dst_path, max_volume):
    f = wave.open(src_path,'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)

    wave_data = norm_by_am(wave_data, max_volume)

    save_f = wave.open(dst_path,'wb')
    save_f.setnchannels(nchannels)
    save_f.setsampwidth(sampwidth)
    save_f.setframerate(framerate)
    save_f.writeframes(wave_data.tostring())
    save_f.close()


if __name__ == '__main__':
    path = '\\small_scale_example\\'
    noise_path = path + '_background_noise_\\'
    dst_path = path + 'time_stretch\\_background_noise_\\'
    file_list = os.listdir(noise_path)
    count = 1
    for file in file_list:
        src = noise_path + file
        dst = dst_path + 'bgn_%d.wav'%(count)
        norm_wave(src, dst, 4000)
        count += 1
