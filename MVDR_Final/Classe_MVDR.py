import numpy as np
from scipy.fftpack import fft
import funcoes

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
    
    def vetor_regula_fase(self, direcao_desejada):
        direcao_desejada = -direcao_desejada
        frequency_vector = np.linspace(0, self.sampling_frequency, self.fft_window)
        qtd_mics = len(self.mic_angle_vector)
        Regula_fases = np.ones((len(frequency_vector), qtd_mics), dtype=np.complex64)

        for f, frequency in enumerate(frequency_vector):
            for m, mic_angle in enumerate(self.mic_angle_vector):
                Regula_fases[f, m] = complex(np.exp(( - 1j) * ((2 * np.pi * frequency) / self.sound_speed)* (self.mic_diameter / 2)* np.sin(np.deg2rad(direcao_desejada) - np.deg2rad(mic_angle))))
        Regula_fases = np.conjugate(Regula_fases).T
        normaliza_Regula_fases = self.normaliza(Regula_fases)
        return normaliza_Regula_fases[:, 0:int(self.fft_window / 2) + 1]  

    def normaliza(self, Regula_fases):
        for j in range(0, self.fft_window):
            weight = np.matmul(np.conjugate(Regula_fases[:, j]).T, Regula_fases[:, j])
            Regula_fases[:, j] = (Regula_fases[:, j] / weight) 
        return Regula_fases
    
    def Matriz_CorrelacaoEspacial(self, multi_signal, use_number_of_frames_init=10, use_number_of_frames_final=10):

        start_index = 0
        end_index = start_index + self.fft_window
        qtd_mics = len(self.mic_angle_vector)
        frequency_grid = np.linspace(0, self.sampling_frequency, self.fft_window)
        frequency_grid = frequency_grid[0:np.int(self.fft_window / 2) + 1]
        speech_length, number_of_channels = np.shape(multi_signal)
        R_mean = np.zeros((qtd_mics, qtd_mics, len(frequency_grid)), dtype=complex)
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
    
    def MVDR_BF(self, Regula_fases, R):
        qtd_mics = len(self.mic_angle_vector)
        frequency_grid = np.linspace(0, self.sampling_frequency, self.fft_window)
        frequency_grid = frequency_grid[0:np.int(self.fft_window / 2) + 1]        
        beamformer = np.ones((qtd_mics, len(frequency_grid)), dtype=np.complex64)
        for f in range(0, len(frequency_grid)):
            R_cut = np.reshape(R[:, :, f], [qtd_mics, qtd_mics])
            inv_R = np.linalg.pinv(R_cut)
            a = np.matmul(np.conjugate(Regula_fases[:, f]).T, inv_R)
            b = np.matmul(a, Regula_fases[:, f])
            b = np.reshape(b, [1, 1])
            beamformer[:, f] = np.matmul(inv_R, Regula_fases[:, f])/b      
        return beamformer
    
    def aplica_beamformer(self, beamformer, spectrum):
        canais, number_of_frames, number_of_bins = np.shape(spectrum)        
        espectro_melhorado = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            espectro_melhorado[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, spectrum[:, :, f])
        return funcoes.spec2wav(espectro_melhorado, self.sampling_frequency, self.fft_window, self.fft_window, self.fft_shift)