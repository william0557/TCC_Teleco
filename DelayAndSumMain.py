import soundfile as sf
import numpy as np
from beamformer import util
from beamformer import delayandsum as ds
import matplotlib.pyplot as pl

input_SoundAngles = np.array([0, 60, 120, 180, 270, 330])
desired_direction = 120
mic_radius = 0.0922
mic_diameter = 2*mic_radius
fft_window = 1024
fft_shift = 512
sampling_freq = 16000

def input_read(input_path=r'./inputs/Mic{N}_440Hz.wav',channel_index=np.array([1, 2, 3, 4, 5, 6])):
    Input_WaveNames = input_path.replace('{N}', str(channel_index[0]))
    wav, _ = sf.read(Input_WaveNames)
    channels_wav = np.zeros((len(wav), len(channel_index)), dtype=np.float32)
    channels_wav[:, 0] = wav
    for i in range(1, len(channel_index)):
        channels_wav[:, i] = sf.read(Input_WaveNames)[0]
    return channels_wav

input_arrays = input_read()

out_fftSpectrum, _ = util.get_3dim_spectrum_from_data(input_arrays, fft_window, fft_shift, fft_window)

ds_beamformer = ds.delayandsum(input_SoundAngles, mic_diameter, sampling_frequency=sampling_freq, fft_window=fft_window, fft_shift=fft_shift)

beamformer = ds_beamformer.get_vetor_direcao(desired_direction)

out_sound = ds_beamformer.aplica_beamformer(beamformer, out_fftSpectrum)

out_path = './output/delaysum_out'+str(desired_direction)+'.wav'
sf.write(out_path, out_sound / np.max(np.abs(out_sound)) * 0.65, sampling_freq)

pl.figure()
pl.plot(out_sound)
