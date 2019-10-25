"""
 audio analysis tool set
"""

import numpy as np
from scipy.io import wavfile
from pathlib import Path
import os
import matplotlib.pyplot as plt
import scipy.io as sio


Fs=16000


frame_len=int((6.25*5/1000)*Fs) # 100 len
erg_TH=frame_len*15 # energy threshold used to seperate noise and speech, assuming data is int16

HIST_FRAME_NUM=10 #number of frames used to track history of frame energy

noise_amp_ave=0

"""
 search for the beginning of sound by checking energy level
 search starts from 'search_start_idx'
 return the start index relative to data[0]
"""
def seek_clip_start(data, search_start_idx):
    frame_idx=0
    FLAG_ST_FOUND=0
    erg_hist=np.zeros(HIST_FRAME_NUM)
    total_frames=int(len(data)-frame_len)

    while (frame_idx < total_frames-1) and (FLAG_ST_FOUND==0):

        tmp=data[search_start_idx+frame_idx*frame_len:search_start_idx+(frame_idx+1)*frame_len]
        tmp=np.abs(tmp)
        erg=np.sum(tmp)
        
        if erg>erg_TH:
            if (np.sum(erg_hist)==0): #sum(erg_hist)==0 indicates silence in previous HIST_FRAME_NUM frames
                FLAG_ST_FOUND=1
                seg_start_idx=search_start_idx+frame_idx*frame_len
        else:
            erg_hist[0:HIST_FRAME_NUM-1]=erg_hist[1:HIST_FRAME_NUM]
            erg_hist[HIST_FRAME_NUM-1]=0
        
        frame_idx+=1

    if FLAG_ST_FOUND==0:
        return None
    else: 
        return seg_start_idx

"""
 search for the end of sound by checking energy level
 search starts from 'search_start_idx'
 return the ending index relative to data[0]
"""

def seek_clip_end(data,search_start_idx):

    frame_idx=0
    FLAG_END_FOUND=0
    erg_hist=np.ones(HIST_FRAME_NUM)
    total_frames=int(len(data)-frame_len)
    
    while (frame_idx < total_frames-1) and (FLAG_END_FOUND==0):

        tmp=data[search_start_idx+frame_idx*frame_len:search_start_idx+(frame_idx+1)*frame_len]
        tmp=np.abs(tmp)
        erg=np.sum(tmp)

        erg_hist[0:HIST_FRAME_NUM-1]=erg_hist[1:HIST_FRAME_NUM]
        
        if erg<erg_TH:
            erg_hist[HIST_FRAME_NUM-1]=0
        else:
            erg_hist[HIST_FRAME_NUM-1]=1

        if (np.sum(erg_hist)<5):
            FLAG_END_FOUND=1
            seg_end_idx=search_start_idx+frame_idx*frame_len
        
        frame_idx+=1

    if FLAG_END_FOUND==0:
        seg_end_idx=search_start_idx+frame_idx*frame_len
    
    return seg_end_idx


#trim or extend data length to fixed length 
def trim_to_fixed_len(data, num_of_seconds):

    data_len=len(data)
    target_len=Fs*num_of_seconds
    stride_len=int(5/1000*Fs) 

    if (data_len>target_len): #data needs to be trimmed, keep the data of target_len with largest energy

        num_of_frames=int((data_len-target_len)/stride_len)

        max_val=0
        max_idx=0
        for i in range(num_of_frames):
            tmp=data[i*stride_len:i*stride_len+target_len]
            tmp_erg=np.mean(np.abs(tmp))
            if tmp_erg>max_val:
                max_val=tmp_erg
                max_idx=i*stride_len
        return data[max_idx:max_idx+target_len]

    elif (data_len==target_len):
        return data

    else: #data is shorter then target_len, padding zeros in the beginning
        st_idx=int((target_len-data_len)/2)
        tmp=np.zeros(target_len,np.int16)
        tmp[st_idx:st_idx+data_len]=data
        return tmp


def trim_to_fix_length_with_shift(data,num_of_seconds,wav_filename,FLAG_ADD_NOISE,FLAG_ADD_REVERB):

    data_len=len(data)
    target_len=Fs*num_of_seconds
    

    if (data_len>=target_len):#data needs to be trimmed, keep the data of target_len with largest energy
    
        if data_len==target_len:
            x=data
        else: #search the segment with largest energy
            stride_len=int(5/1000*Fs)        
            num_of_frames=int((data_len-target_len)/stride_len)

            max_val=0
            max_idx=0
            
            for i in range(num_of_frames):
                tmp=data[i*stride_len:i*stride_len+target_len]
                tmp_erg=np.mean(np.abs(tmp))
                if tmp_erg>max_val:
                    max_val=tmp_erg
                    max_idx=i*stride_len

            x=data[max_idx:max_idx+target_len]
        
        out_file=wav_filename
        if FLAG_ADD_REVERB:
            for room_id,rir in RIR.items():
                out_file=wav_filename.split('.wav')[0]+'_'+room_id+'.wav'
                x_rev=add_reverb(x,rir)
                if FLAG_ADD_NOISE:
                    out_file=out_file.split('.wav')[0]+'_'+noise['type']+'_%ddB.wav' % int(noise['SNR'])
                    x_rev=add_noise(x_rev)
                write_to_wav_file(x_rev,out_file)
        else:     
           if FLAG_ADD_NOISE:
                x=add_noise(x)
                out_file=out_file.split('.wav')[0]+'_'+noise['type']+'_%ddB.wav' % int(noise['SNR'])
           write_to_wav_file(x,out_file)


    else: #data_len <target_len
        
        #add multiples of shift_ms to data
        shift_ms=100
        stride_len=int(shift_ms/1000*Fs)
        num_of_shifts=int((target_len-data_len)/stride_len)+1

        
        for i in range(num_of_shifts):
            x=np.zeros(target_len,np.int16)
            x[i*stride_len:i*stride_len+data_len]=data
            new_filename=wav_filename.split('.wav')[0]+'_shift%dms.wav' % int(i*shift_ms)
            if 0:
                plt.figure(1)
                plt.plot(x)
                plt.ylim(-2**15,2**15)
                plt.title('original')
                plt.show()

            out_file=new_filename
            if FLAG_ADD_REVERB:
                for room_id,rir in RIR.items():
                    out_file=new_filename.split('.wav')[0]+'_'+room_id+'.wav'
                    x_rev=add_reverb(x,rir)
                    if 0:
                        plt.figure(2)
                        plt.plot(x_rev)
                        plt.ylim(-2**15,2**15)
                        plt.title('reverb')
                        plt.show()

                    if FLAG_ADD_NOISE:
                        out_file=out_file.split('.wav')[0]+'_'+noise['type']+'_%ddB.wav' % int(noise['SNR'])
                        x_rev=add_noise(x_rev)
                    write_to_wav_file(x_rev,out_file)
            else:     
               if FLAG_ADD_NOISE:
                    x=add_noise(x)
                    out_file=out_file.split('.wav')[0]+'_'+noise['type']+'_%ddB.wav' % int(noise['SNR'])
               write_to_wav_file(x,out_file)


def write_to_wav_file(data_int16, WAVE_OUTPUT_FILENAME):
    wavfile.write(WAVE_OUTPUT_FILENAME,Fs,np.array(data_int16).astype(np.int16))

def prepare_noise(noise_type,SNR):
    global noise

    if noise_type=='pink':
        Fs,data=wavfile.read('pink_noise_60s.wav')

    noise_amp_ave=np.mean(np.abs(data[0:Fs-1]))

    noise={'type': noise_type,
                'SNR': SNR,
                'data':data,
                'amp_ave':noise_amp_ave}


def add_noise(clean_data):

    data_amp_ave=np.mean(np.abs(clean_data))
    noise_scale=(data_amp_ave/noise['amp_ave'])*10**(-noise['SNR']/20)
    data_len=len(clean_data)

    #select a random segment of noise to mix
    num_noise_seg= np.floor(noise['data']/data_len)
    noise_seg=np.random.randint(0,num_noise_seg,1)
    
    tmp=noise_scale*noise['data'][int(noise_seg*data_len):int((noise_seg+1)*data_len)]

    noisy_data=clean_data+tmp
    
    #prevent overflow
    scale2=2**15/np.max(np.abs(noisy_data))
    if scale2<1:
        noisy_data=np.fix(noisy_data*scale2).astype('int16')

    return noisy_data

def prepare_reverb(RIRs_dir):
    
    global RIR
    
    RIR={}

    for file in os.listdir(RIRs_dir):
        if file.endswith('.mat'):
            print(file)
            room_id=file.split('_')[0]

            rir_filename=os.path.join(RIRs_dir, file)
            var=sio.loadmat(rir_filename)
            h=var['RIR_cell'].flatten()[0]
            #RIR[room_id]=np.flip(h,axis=0).flatten()
            RIR[room_id]=h.flatten()
            #RIR['h%d'%rir_count]=h

#add reverb
def add_reverb(x,rir):
        
    tmp_x=np.zeros(len(x)+len(rir))
    tmp_x[len(rir):]=x
    #y=convolve1d(tmp_x,rir,mode='constant') #cut off beginning of signal, don't use
    tmp=np.convolve(x,rir,mode='full')
    
    y=tmp[0:len(x)] #keep the length the same

    #adjust energy of y to the same as x
    x_amp_ave=np.mean(np.abs(x))
    y_amp_ave=np.mean(np.abs(y))
    scale=x_amp_ave/y_amp_ave
    y=y*scale

    #prevent overflow
    scale2=2**15/np.max(np.abs(y))
    if scale2<1:
        y=y*scale2

    y=np.fix(y).astype('int16')

    return y
