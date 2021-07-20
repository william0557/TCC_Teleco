# -*- coding: utf-8 -*-
import numpy as np
from . import util

class delayandsum:
    
    def __init__(self, mic_angle_vector,mic_diameter,sound_speed=343,sampling_frequency=41000,fft_window=1024,fft_shift=512):
        self.fft_window=fft_window
        self.fft_shift=fft_shift
        self.mic_angle_vector=mic_angle_vector
        self.mic_diameter=mic_diameter
        self.sound_speed=sound_speed
        self.sampling_frequency=sampling_frequency

    def normaliza(self, vetor_direcao):        
        for i in range(0, self.fft_window):            
            weight = np.matmul(np.conjugate(vetor_direcao[:, i]).T, vetor_direcao[:, i])
            vetor_direcao[:, i] = (vetor_direcao[:, i] / weight) 
        return vetor_direcao        
    
    def get_vetor_direcao(self, direcao_desejada):
        frequency_vector = np.linspace(0, self.sampling_frequency, self.fft_window)
        number_of_mic = len(self.mic_angle_vector)
        vetor_direcao = np.ones((len(frequency_vector), number_of_mic), dtype=np.complex64)
        direcao_desejada = direcao_desejada * (-1)
        for f, frequency in enumerate(frequency_vector):
            for m, mic_angle in enumerate(self.mic_angle_vector):
                vetor_direcao[f, m] = np.complex(np.exp(( - 1j) * ((2 * np.pi * frequency) / self.sound_speed)* (self.mic_diameter / 2)* np.cos(np.deg2rad(direcao_desejada) - np.deg2rad(mic_angle))))
        vetor_direcao = np.conjugate(vetor_direcao).T
        normaliza_vetor_direcao = self.normaliza(vetor_direcao)
        return normaliza_vetor_direcao[:, 0:np.int(self.fft_window / 2) + 1]    
    
    def aplica_beamformer(self, beamformer, spectrum):
        canais, number_of_frames, number_of_bins = np.shape(spectrum)        
        espectro_melhorado = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            espectro_melhorado[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, spectrum[:, :, f])
        return util.spec2wav(espectro_melhorado, self.sampling_frequency, self.fft_window, self.fft_window, self.fft_shift)