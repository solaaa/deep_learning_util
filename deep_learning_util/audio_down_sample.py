import librosa
import soundfile as sf
import os

def wav_file_resample(src, dst, dst_sample):
    """
    对目标文件进行降采样，采样率为dst_sample
    :param src:源文件路径
    :param dst:降采样后文件保存路径
    :param dst_sample:降采样后的采样率
    :return:
    """
    src_sig, sr = sf.read(src)
    dst_sig = librosa.resample(src_sig, sr, dst_sample)
    sf.write(dst, dst_sig, dst_sample)

if __name__ == '__main__':
    sample_rate = 16000
    path = './data/noisy_trainset_wav_16k/'
    path_list = os.listdir(path)
    print('start' + '-'*30)
    for file_name in path_list:
        src = path + file_name
        dst = path + file_name 
        wav_file_resample(src, dst, sample_rate)
    print('finish' + '-'*30)