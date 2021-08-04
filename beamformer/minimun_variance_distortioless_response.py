import numpy as np
from scipy.fftpack import fft
from . import funcoes

class Superdirective:
    
    def __init__(self,
                 mic_angle_vector,
                 mic_diameter,
                 sampling_frequency=41000,
                 fft_window=512,
                 fft_shift=128,
                 sound_speed=343):  


        self.sound_speed=sound_speed  
        self.mic_angle_vector=mic_angle_vector
        self.mic_diameter=mic_diameter
        self.sampling_frequency=sampling_frequency
        self.fft_window=fft_window
        self.fft_shift=fft_shift
    
    def get_vetor_direcao(self, direcao_desejada):
        direcao_desejada = -direcao_desejada
        frequency_vector = np.linspace(0, self.sampling_frequency, self.fft_window)
        number_of_mic = len(self.mic_angle_vector)
        vetor_direcao = np.ones((len(frequency_vector), number_of_mic), dtype=np.complex64)

        for f, frequency in enumerate(frequency_vector):
            for m, mic_angle in enumerate(self.mic_angle_vector):
                vetor_direcao[f, m] = np.complex(np.exp(( - 1j) * ((2 * np.pi * frequency) / self.sound_speed)* (self.mic_diameter / 2)* np.cos(np.deg2rad(direcao_desejada) - np.deg2rad(mic_angle))))
        vetor_direcao = np.conjugate(vetor_direcao).T
        normaliza_vetor_direcao = self.normaliza(vetor_direcao)
        return normaliza_vetor_direcao[:, 0:np.int(self.fft_window / 2) + 1]
    
    def normaliza(self, vetor_direcao):
        for j in range(0, self.fft_window):
            weight = np.matmul(np.conjugate(vetor_direcao[:, j]).T, vetor_direcao[:, j])
            vetor_direcao[:, j] = (vetor_direcao[:, j] / weight) 
        return vetor_direcao        
    
    def Matriz_CorrelacaoEspacial(self, multi_signal, use_number_of_frames_init=10, use_number_of_frames_final=10):

        start_index = 0
        end_index = start_index + self.fft_window
        number_of_mic = len(self.mic_angle_vector)
        frequency_grid = np.linspace(0, self.sampling_frequency, self.fft_window)
        frequency_grid = frequency_grid[0:np.int(self.fft_window / 2) + 1]
        speech_length, number_of_channels = np.shape(multi_signal)
        R_mean = np.zeros((number_of_mic, number_of_mic, len(frequency_grid)), dtype=np.complex64)
        used_number_of_frames = 0
        
        # forward
        for _ in range(0, use_number_of_frames_init):
            multi_signal_cut = multi_signal[start_index:end_index, :]
            complex_signal = fft(multi_signal_cut, n=self.fft_window, axis=0)
            for f in range(0, len(frequency_grid)):
                    R_mean[:, :, f] = R_mean[:, :, f] + np.multiply.outer(complex_signal[f, :], np.conj(complex_signal[f, :]).T)
            used_number_of_frames = used_number_of_frames + 1
            start_index = start_index + self.fft_shift
            end_index = end_index + self.fft_shift
            if speech_length <= start_index or speech_length <= end_index:
                used_number_of_frames = used_number_of_frames - 1
                break
        
        # backward
        end_index = speech_length
        start_index = end_index - self.fft_window
        for _ in range(0, use_number_of_frames_final):
            multi_signal_cut = multi_signal[start_index:end_index, :]
            complex_signal = fft(multi_signal_cut, n=self.fft_window, axis=0)
            for f in range(0, len(frequency_grid)):
                R_mean[:, :, f] = R_mean[:, :, f] + np.multiply.outer(complex_signal[f, :], np.conj(complex_signal[f, :]).T)
            used_number_of_frames = used_number_of_frames + 1
            start_index = start_index - self.fft_shift
            end_index = end_index - self.fft_shift
            if  start_index < 1 or end_index < 1:
                used_number_of_frames = used_number_of_frames - 1
                break

        return R_mean / used_number_of_frames  
    
    def MVDR_BF(self, vetor_direcao, R):
        number_of_mic = len(self.mic_angle_vector)
        frequency_grid = np.linspace(0, self.sampling_frequency, self.fft_window)
        frequency_grid = frequency_grid[0:np.int(self.fft_window / 2) + 1]        
        beamformer = np.ones((number_of_mic, len(frequency_grid)), dtype=np.complex64)
        for f in range(0, len(frequency_grid)):
            R_cut = np.reshape(R[:, :, f], [number_of_mic, number_of_mic])
            inv_R = np.linalg.pinv(R_cut)
            a = np.matmul(np.conjugate(vetor_direcao[:, f]).T, inv_R)
            b = np.matmul(a, vetor_direcao[:, f])
            b = np.reshape(b, [1, 1])
            beamformer[:, f] = np.matmul(inv_R, vetor_direcao[:, f])/b      
        return beamformer
    
    def aplica_beamformer(self, beamformer, spectrum):
        canais, number_of_frames, number_of_bins = np.shape(spectrum)        
        espectro_melhorado = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            espectro_melhorado[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, spectrum[:, :, f])
        return funcoes.spec2wav(espectro_melhorado, self.sampling_frequency, self.fft_window, self.fft_window, self.fft_shift)