import numpy as np
from beamformer import funcoes
from beamformer import minimum_variance_distortioless_response as mvdr
import soundfile as sf
import matplotlib.pyplot as pl
import time

start_time = time.time()

input_SoundAngles = np.array([300, 240, 180, 120, 60, 0])
desired_direction = 300
mic_radius = 0.0461
mic_diameter = 2*mic_radius
fft_window = 512
fft_shift = 128
sampling_freq = 44100


def input_read(input_path=r'./inputs/mic{N}.wav',channel_index=np.array([1, 2, 3, 4, 5, 6])):
    Input_WaveNames = input_path.replace('{N}', str(channel_index[0]))
    wav, _ = sf.read(Input_WaveNames)
    channels_wav = np.zeros((len(wav), len(channel_index)), dtype=np.float32)
    channels_wav[:, 0] = wav
    for i in range(1, len(channel_index)):
        channels_wav[:, i] = sf.read(Input_WaveNames)[0]
    return channels_wav

input_arrays = input_read()

complex_spectrum, _ = funcoes.espectro_3D(input_arrays, fft_window, fft_shift, fft_window)

mvdr_beamformer = mvdr.Superdirective(input_SoundAngles, mic_diameter, sampling_frequency=sampling_freq, fft_window=fft_window, fft_shift=fft_shift)

vetor_direcao = mvdr_beamformer.get_vetor_direcao(desired_direction)

Correlacao_espacial = mvdr_beamformer.Matriz_CorrelacaoEspacial(input_arrays)

beamformer = mvdr_beamformer.MVDR_BF(vetor_direcao, Correlacao_espacial)

out_sound = mvdr_beamformer.aplica_beamformer(beamformer, complex_spectrum)

out_path = './output/MVDR_Out_'+str(desired_direction)+'.wav'

sf.write(out_path, out_sound / np.max(np.abs(out_sound)) * 0.65, sampling_freq)

print("%s seconds" %(time.time()-start_time))

dt = 1/sampling_freq
t = np.arange(0, 1783936/44100, dt)

fourier_transform = np.fft.rfft(out_sound)

abs_fourier_transform = np.abs(fourier_transform)

power_spectrum = np.square(abs_fourier_transform)

# Plots

t = np.linspace(0, sampling_freq/2, len(power_spectrum))
pl.xlim(0, 5000)
pl.plot(t, power_spectrum)
pl.show()