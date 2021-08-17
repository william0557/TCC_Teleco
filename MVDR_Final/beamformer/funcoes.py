import numpy as np
import soundfile as sf
from scipy import signal as sg
from scipy.fftpack import fft, ifft

def espectro_3D(wav_data, frame, fft_shift, fft_window):
    
    len_sample, len_channel_vec = np.shape(wav_data)
    dump_wav = wav_data.T
    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    start = 0
    end = frame
    number_of_frame = int((len_sample - frame)/fft_shift)
    spectrums = np.zeros((len_channel_vec, number_of_frame, int(fft_window / 2) + 1), dtype=np.complex64)
    for i in range(0, number_of_frame):       
        multi_signal_spectrum = fft(dump_wav[:, start:end], n=fft_window, axis=1)[:, 0:int(fft_window / 2) + 1]       
        spectrums[:, i, :] = multi_signal_spectrum
        end = end + fft_shift
        start = start + fft_shift
    return spectrums, len_sample

def spec2wav(spectrogram, sampling_frequency, fft_window, frame_len, fft_shift):

    n_of_frame, a = np.shape(spectrogram)    
    hanning = sg.hanning(fft_window + 1, 'periodic')[: - 1]    
    cut_data = np.zeros(fft_window, dtype=np.complex64)
    result = np.zeros(300*sampling_frequency, dtype=np.float32)
    start_point = 0
    end_point = start_point + frame_len
    for ii in range(0, n_of_frame):
        half_spec = spectrogram[ii, :]        
        cut_data[0:np.int(fft_window / 2) + 1] = half_spec.T   
        cut_data[np.int(fft_window / 2) + 1:] =  np.flip(np.conjugate(half_spec[1:np.int(fft_window / 2)]), axis=0)
        cut_data2 = np.real(ifft(cut_data, n=fft_window))        
        result[start_point:end_point] = result[start_point:end_point] + np.real(cut_data2 * hanning.T) 
        start_point = start_point + fft_shift
        end_point = end_point + fft_shift

    Out_wave = result[0:end_point - fft_shift]

    return Out_wave