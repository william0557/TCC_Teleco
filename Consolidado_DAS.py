import soundfile as sf
import numpy as np
from scipy import signal as sg
from scipy.fftpack import fft, ifft
import time

start_time = time.time()

input_SoundAngles = np.array([300, 240, 180, 120, 60, 0])
desired_direction = 300
mic_radius = 0.0461
mic_diameter = 2*mic_radius
fft_window = 512
fft_shift = 128
sampling_freq = 44100

global sound_speed

sound_speed = 343

def input_read(input_path=r'./inputs/mic{N}.wav',channel_index=np.array([1, 2, 3, 4, 5, 6])):
    Input_WaveNames = input_path.replace('{N}', str(channel_index[0]))
    wav, _ = sf.read(Input_WaveNames)
    channels_wav = np.zeros((len(wav), len(channel_index)), dtype=np.float32)
    channels_wav[:, 0] = wav
    for i in range(1, len(channel_index)):
        channels_wav[:, i] = sf.read(Input_WaveNames)[0]
    return channels_wav

def normaliza(vetor_direcao):        
        for i in range(0, fft_window):            
            weight = np.matmul(np.conjugate(vetor_direcao[:, i]).T, vetor_direcao[:, i])
            vetor_direcao[:, i] = (vetor_direcao[:, i] / weight) 
        return vetor_direcao        
    
def get_phaseshifts(direcao_desejada, mic_angle_vector): 
    
    frequency_vector = np.linspace(0, sampling_freq, fft_window)
    number_of_mic = len(mic_angle_vector)
    phase_shifts = np.ones((len(frequency_vector), number_of_mic), dtype=complex)
    direcao_desejada = direcao_desejada * (-1)
    for f, frequency in enumerate(frequency_vector):
        for m, mic_angle in enumerate(mic_angle_vector):
            phase_shifts[f, m] = complex(np.exp(( - 1j) * ((2 * np.pi * frequency) / sound_speed)* (mic_diameter / 2)* np.cos(np.deg2rad(direcao_desejada) - np.deg2rad(mic_angle))))
    phase_shifts = np.conjugate(phase_shifts).T
    normaliza_phaseshifts = normaliza(phase_shifts)
    return normaliza_phaseshifts[:, 0:int(fft_window / 2) + 1]    

def aplica_beamformer(beamformer, spectrum):
    canais, number_of_frames, number_of_bins = np.shape(spectrum)        
    espectro_melhorado = np.zeros((number_of_frames, number_of_bins), dtype=complex)
    for f in range(0, number_of_bins):
        espectro_melhorado[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, spectrum[:, :, f])
    return spec2wav(espectro_melhorado, sampling_freq, fft_window, fft_window, fft_shift)

def espectro_3D(wav_data, frame, fft_shift, fft_window):
    
    len_sample, len_channel_vec = np.shape(wav_data)
    dump_wav = wav_data.T
    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    start = 0
    end = frame
    number_of_frame = int((len_sample - frame)/fft_shift)
    spectrums = np.zeros((len_channel_vec, number_of_frame, int(fft_window / 2) + 1), dtype=complex)

    for i in range(0, number_of_frame):       
        multi_signal_spectrum = fft(dump_wav[:, start:end], n=fft_window, axis=1)[:, 0:int(fft_window / 2) + 1]       
        spectrums[:, i, :] = multi_signal_spectrum
        end = end + fft_shift
        start = start + fft_shift
    return spectrums, len_sample

def spec2wav(spectrogram, sampling_freq, fft_window, frame_len, fft_shift):

    n_of_frame, a = np.shape(spectrogram)    
    janela_hann = sg.windows.hann(fft_window + 1, 'periodic')[: - 1]    
    cut_data = np.zeros(fft_window, dtype=np.complex64)
    result = np.zeros(300*sampling_freq, dtype=np.float32)
    start_point = 0
    end_point = start_point + frame_len
    for ii in range(0, n_of_frame):
        half_spec = spectrogram[ii, :]        
        cut_data[0:int(fft_window / 2) + 1] = half_spec.T   
        cut_data[int(fft_window / 2) + 1:] =  np.flip(np.conjugate(half_spec[1:int(fft_window / 2)]), axis=0)
        cut_data2 = np.real(ifft(cut_data, n=fft_window))        
        result[start_point:end_point] = result[start_point:end_point] + np.real(cut_data2 * janela_hann.T) 
        start_point = start_point + fft_shift
        end_point = end_point + fft_shift

    Out_wave = result[0:end_point - fft_shift]

    return Out_wave

input_arrays = input_read()

out_fftSpectrum, _ = espectro_3D(input_arrays, fft_window, fft_shift, fft_window)

beamformer = get_phaseshifts(desired_direction, input_SoundAngles)

print(beamformer[:,1]) #delays

out_sound = aplica_beamformer(beamformer, out_fftSpectrum)

out_path = './output/delayandsumOut_'+str(desired_direction)+'.wav'

sf.write(out_path, out_sound / np.max(np.abs(out_sound)) * 0.65, sampling_freq)
print("%s seconds" %(time.time()-start_time))