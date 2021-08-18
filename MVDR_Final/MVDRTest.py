import numpy as np
import time
import soundfile as sf
import matplotlib.pyplot as plt
import funcoes
import Classe_MVDR as mvdrbr

Fs = 44100
fft_window = 512
fft_shift = 256

Angulos_microfones = np.array([300, 240, 180, 120, 60, 0])

Diametro_rede = 0.0461*2

def aquisicao_sinais(input_path=r'./sample_data/mic{}.wav', indice_vetor=np.array([1, 2, 3, 4, 5, 6])):
        wav, _ = sf.read(input_path.replace('{}', str(indice_vetor[0])), dtype='float32')
        ondas_sonoras = np.zeros((len(wav), len(indice_vetor)), dtype=np.float32)
        ondas_sonoras[:, 0] = wav
        for i in range(1, len(indice_vetor)):
            ondas_sonoras[:, i] = sf.read(input_path.replace('{}', str(indice_vetor[i])), dtype='float32')[0]
        return ondas_sonoras
# testa para cara direção
for i in Angulos_microfones:
    direcao_desejada = i
    out_path = './output/Direção_'+str(i)+'2000x800Hz_mvdr.wav'
    
    start_time = time.time()

    mvdr_beamformer = mvdrbr.Superdirective(Angulos_microfones, Diametro_rede, sampling_frequency=Fs, fft_window=fft_window, fft_shift=fft_shift)
    
    regula_fases = mvdr_beamformer.vetor_regula_fase(direcao_desejada)

    ondas_sonoras = aquisicao_sinais()

    Correlacao_espacial = mvdr_beamformer.Matriz_CorrelacaoEspacial(ondas_sonoras, 10, 10)
    bf = mvdr_beamformer.MVDR_BF(regula_fases, Correlacao_espacial)
    espectro, _ = funcoes.espectro_3D(ondas_sonoras, fft_window, fft_shift, fft_window)

    out_sound = mvdr_beamformer.aplica_beamformer(bf, espectro)
    
    print("%s seconds" %(time.time()-start_time))

    sf.write(out_path, out_sound / np.max(np.abs(out_sound)) * 0.65, Fs)

    # Plots
    Nfft = 10000
    Y1 = np.fft.fft(out_sound,Nfft)
    Y2 = np.fft.fft(ondas_sonoras[:,1],Nfft)
    w = np.arange(0,2*np.pi, 2*np.pi/Nfft)
    f = (Fs/(2*np.pi))*w
    modY1 = 10*np.log10(abs(Y1))
    angleY1 = np.angle(Y1)

    modY2 = 10*np.log10(abs(Y2))
    angleY2 = np.angle(Y2)

    plt.plot(f[0:500],modY1[0:500])
    plt.title('Após MVDR direção ' + str(i)+'º')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)
    plt.figure()

    plt.plot(f[0:500],modY2[0:500])
    plt.title('Antes do MVDR')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)
    plt.figure()
