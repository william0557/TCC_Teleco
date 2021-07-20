import numpy as np
import soundfile as sf
import numpy.matlib as npm
from scipy import signal as sg
from scipy.fftpack import fft, ifft

def stab(mat, theta, num_channels):
    d = np.power(np.array(10, dtype=np.complex64) , np.arange( - num_channels, 0, dtype=np.float))
    result_mat = mat
    for i in range(1, num_channels + 1):
        if np.linalg.cond(mat) > theta:
            return result_mat
        result_mat = result_mat + d[i - 1] * np.eye(num_channels, dtype=np.complex64)
    return result_mat

def get_3dim_spectrum(wav_name, channel_vec, start_point, stop_point, frame, shift, fftl):
    """
    dump_wav : channel_size * speech_size (2dim)
    """
    samples, _ = sf.read(wav_name.replace('{}', str(channel_vec[0])), start=start_point, stop=stop_point, dtype='float32')
    if len(samples) == 0:
        return None,None
    dump_wav = np.zeros((len(channel_vec), len(samples)), dtype=np.float16)
    dump_wav[0, :] = samples.T
    for ii in range(0,len(channel_vec) - 1):
        samples,_ = sf.read(wav_name.replace('{}', str(channel_vec[ii +1 ])), start=start_point, stop=stop_point, dtype='float32')
        dump_wav[ii + 1, :] = samples.T    

    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    window = sg.hanning(fftl + 1, 'periodic')[: - 1]
    multi_window = npm.repmat(window, len(channel_vec), 1)    
    st = 0
    ed = frame
    number_of_frame = np.int((len(samples) - frame) /  shift)
    spectrums = np.zeros((len(channel_vec), number_of_frame, np.int(fftl / 2) + 1), dtype=np.complex64)
    for ii in range(0, number_of_frame):
        multi_signal_spectrum = fft(dump_wav[:, st:ed], n=fftl, axis=1)[:, 0:np.int(fftl / 2) + 1] # channel * number_of_bin        
        spectrums[:, ii, :] = multi_signal_spectrum
        st = st + shift
        ed = ed + shift
    return spectrums, len(samples)

def get_3dim_spectrum_from_data(wav_data, frame, shift, fft_window):
    
    len_sample, len_channel_vec = np.shape(wav_data)            
    dump_wav = wav_data.T
    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    window = sg.hanning(fft_window + 1, 'periodic')[: - 1]
    multi_window = npm.repmat(window, len_channel_vec, 1)    
    st = 0
    ed = frame
    number_of_frame = np.int((len_sample - frame) /  shift)
    spectrums = np.zeros((len_channel_vec, number_of_frame, np.int(fft_window / 2) + 1), dtype=np.complex64)
    for i in range(0, number_of_frame):       
        multi_signal_spectrum = fft(dump_wav[:, st:ed], n=fft_window, axis=1)[:, 0:np.int(fft_window / 2) + 1] # channel * number_of_bin        
        spectrums[:, i, :] = multi_signal_spectrum
        ed = ed + shift
        st = st + shift
        
    return spectrums, len_sample

def my_det(matrix_):
    sign, lodget = np.linalg.slogdet(matrix_)
    return np.exp(lodget)

def spec2wav(spectrogram, sampling_frequency, fft_window, frame_len, fft_shift):
    n_of_frame, fft_half = np.shape(spectrogram)    
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
    return result[0:end_point - fft_shift]

def multispec2wav(multi_spectrogram, beamformer, fftl, shift, multi_window, true_dur):
    channel, number_of_frame, fft_size = np.shape(multi_spectrogram)
    cut_data = np.zeros((channel, fftl), dtype=np.complex64)
    result = np.zeros((channel, true_dur), dtype=np.float32)
    start_p = 0
    end_p = start_p + fftl
    for ii in range(0, number_of_frame):
        cut_spec = multi_spectrogram[:, ii, :] * beamformer
        cut_data[:, 0:fft_size] = cut_spec
        cut_data[:, fft_size:] = np.transpose(np.flip(cut_spec[:, 1:fft_size - 1], axis=1).T)
        cut_data2 = np.real(ifft(cut_data, n=fftl, axis=1))
        result[:, start_p:end_p] = result[:, start_p:end_p] + (cut_data2 * multi_window)
        start_p = start_p + shift
        end_p = end_p + shift
    return np.sum(result[:,0:end_p - shift], axis=0)
           
        
def check_beamformer(freq_beamformer,theta_cov):
    freq_beamformer = np.real(freq_beamformer)
    if len(freq_beamformer[freq_beamformer>=theta_cov])!=0:
        return np.ones(np.shape(freq_beamformer),dtype=np.complex64) * (1+1j)
    return freq_beamformer


              