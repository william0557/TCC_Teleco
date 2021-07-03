import numpy as np
import soundfile as sf
from beamformer import delayandsum as ds
from beamformer import util

sampling_freq = 16000
out_path = './output/delaysum_out.wav'
input_SoundAngles = np.array([0, 60, 120, 180, 270, 330])
desired_direction = 0
mic_radius = 0.0922
mic_diameter = 2*mic_radius
fft_window = 512
fft_shift = 256

# def multi_channel_read(prefix=r'./sample_data/20G_20GO010I_STR.CH{}.wav',
#                        channel_index_vector=np.array([1, 2, 3, 4, 5, 6])):
#     wav, _ = sf.read(prefix.replace('{}', str(channel_index_vector[0])), dtype='float32')
#     wav_multi = np.zeros((len(wav), len(channel_index_vector)), dtype=np.float32)
#     wav_multi[:, 0] = wav
#     for i in range(1, len(channel_index_vector)):
#         wav_multi[:, i] = sf.read(prefix.replace('{}', str(channel_index_vector[i])), dtype='float32')[0]
#     return wav_multi

#input_arrays = multi_channel_read()  #####################################COLOCAR OS ARRAYS DE ENTRADA AQUI!!!#

out_fftSpectrum, _ = util.get_3dim_spectrum_from_data(input_arrays, fft_window, fft_shift, fft_window)

ds_beamformer = ds.delaysum(input_SoundAngles, mic_diameter, sampling_frequency=sampling_freq, fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)

beamformer = ds_beamformer.get_sterring_vector(desired_direction)

enhanced_speech = ds_beamformer.apply_beamformer(beamformer, out_fftSpectrum)

sf.write(out_path, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65, sampling_freq)
