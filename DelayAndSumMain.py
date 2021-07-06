import soundfile as sf
import numpy as np
from beamformer import util
from beamformer import delayandsum as ds

input_SoundAngles = np.array([0, 60, 120, 180, 270, 330])
desired_direction = 0
mic_radius = 0.0922
mic_diameter = 2*mic_radius
fft_window = 512
fft_shift = 256
sampling_freq = 16000

def input_read(prefix=r'./sample_data/20G_20GO010I_STR.CH{}.wav',
    channel_index=np.array([1, 2, 3, 4, 5, 6])):
    wav, _ = sf.read(prefix.replace('{}', str(channel_index[0])), dtype='float32')
    channels_wav = np.zeros((len(wav), len(channel_index)), dtype=np.float32)
    channels_wav[:, 0] = wav
    for i in range(1, len(channel_index)):
        channels_wav[:, i] = sf.read(prefix.replace('{}', str(channel_index[i])), dtype='float32')[0]
    return channels_wav

input_arrays = input_read()

out_fftSpectrum, _ = util.get_3dim_spectrum_from_data(input_arrays, fft_window, fft_shift, fft_window)

ds_beamformer = ds.delayandsum(input_SoundAngles, mic_diameter, sampling_frequency=sampling_freq, fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)

beamformer = ds_beamformer.get_sterring_vector(desired_direction)

out_sound = ds_beamformer.apply_beamformer(beamformer, out_fftSpectrum)

out_path = './output/delaysum_out'+str(desired_direction)+'.wav'
sf.write(out_path, out_sound / np.max(np.abs(out_sound)) * 0.65, sampling_freq)
