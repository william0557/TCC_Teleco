import threading
import subprocess
import time
import os
import wave
import struct
import sys
import subprocess
import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write

def grava_mic():
    ret = subprocess.run("arecord -D ac108 -d 10 -f S16_LE -r 44100 -c 8 entrada.wav", stdout=subprocess.PIPE, shell=True)
    
def play_mic():
    ret = subprocess.run("aplay -D default saida.wav", stdout=subprocess.PIPE, shell=True)

def direcaochegada(dir_chegada):
#     dir_chegada = 5*np.pi/3

    #Coordenadas esféricas de cada microfone, considerando diâmetro de 9.2
    mic1_esf= (4.61,2*np.pi/3)
    mic2_esf= (4.61,np.pi/3)
    mic3_esf= (4.61,0)
    mic4_esf= (4.61,-np.pi/3)
    mic5_esf= (4.61,-2*np.pi/3)
    mic6_esf= (4.61,np.pi)
    
    #Passagem de coordenadas esféricas para coordenadas cartesianas
    
    mic1_cart = (mic1_esf[0]*np.cos(mic1_esf[1]),mic1_esf[0]*np.sin(mic1_esf[1]))
    mic2_cart = (mic2_esf[0]*np.cos(mic2_esf[1]),mic2_esf[0]*np.sin(mic2_esf[1]))
    mic3_cart = (mic3_esf[0]*np.cos(mic3_esf[1]),mic3_esf[0]*np.sin(mic3_esf[1]))
    mic4_cart = (mic4_esf[0]*np.cos(mic4_esf[1]),mic4_esf[0]*np.sin(mic4_esf[1]))
    mic5_cart = (mic5_esf[0]*np.cos(mic5_esf[1]),mic5_esf[0]*np.sin(mic5_esf[1]))
    mic6_cart = (mic6_esf[0]*np.cos(mic6_esf[1]),mic6_esf[0]*np.sin(mic6_esf[1]))
    
    #definição do mic de chegada
    if dir_chegada <= np.pi/6 or dir_chegada >= 11*np.pi/6:
        mic_chegada = mic3_cart
        mic_saida = mic6_cart
    elif dir_chegada >= np.pi/6 and dir_chegada <= np.pi/2:
        mic_chegada = mic2_cart
        mic_saida = mic5_cart
    elif dir_chegada >= np.pi/2 and dir_chegada <= 5*np.pi/6:
        mic_chegada = mic1_cart
        mic_saida = mic4_cart
    elif dir_chegada >= 5*np.pi/6 and dir_chegada <= 7*np.pi/6:
        mic_chegada = mic6_cart
        mic_saida = mic3_cart
    elif dir_chegada >=7*np.pi/6 and dir_chegada <= 3*np.pi/2:
        mic_chegada = mic5_cart
        mic_saida = mic2_cart
    elif dir_chegada >= 3*np.pi/2 and dir_chegada <= 11*np.pi/6:
        mic_chegada = mic4_cart
        mic_saida = mic1_cart
        
    #Inclinação da onda plana de chegada
    if (dir_chegada >= 0 and dir_chegada <= np.pi/2):
        inclinacao = np.pi/2 + dir_chegada
    if (dir_chegada >= np.pi/2 and dir_chegada <= np.pi):
        inclinacao = -np.pi/2 + dir_chegada
    if (dir_chegada >= np.pi and dir_chegada <= 3*np.pi/2):
        inclinacao = np.pi/2 + dir_chegada - np.pi
    if  (dir_chegada >= 3*np.pi/2 and dir_chegada <= 4*np.pi/2):
        inclinacao = -np.pi/2 + dir_chegada - np.pi
    
    #com a inclinação da frente de onda e com o mic de chegada, podemos traçar a reta que determina a frente de onda plana
    a = np.tan(inclinacao)
    b = -1
    c = mic_saida[1] - mic_saida[0]*np.tan(inclinacao)
    
    #criar casos especiais para chegada de 0 graus e 180 graus 
    #distancias em relação ao microfone de saída em centimetros
    if dir_chegada == 0:
        distancia_1 = 2.305
        distancia_2 = 6.915
        distancia_3 = 9.22
        distancia_4 = 6.915
        distancia_5 = 2.305
        distancia_6 = 0
    elif dir_chegada == np.pi:
        distancia_1 = 6.915
        distancia_2 = 2.305
        distancia_3 = 0
        distancia_4 = 2.305
        distancia_5 = 6.915
        distancia_6 = 9.22
    else:
        distancia_1 = np.abs(a*mic1_cart[0] + b*mic1_cart[1] + c)/(np.sqrt(a**2+b**2))
        distancia_2 = np.abs(a*mic2_cart[0] + b*mic2_cart[1] + c)/(np.sqrt(a**2+b**2))
        distancia_3 = np.abs(a*mic3_cart[0] + b*mic3_cart[1] + c)/(np.sqrt(a**2+b**2))
        distancia_4 = np.abs(a*mic4_cart[0] + b*mic4_cart[1] + c)/(np.sqrt(a**2+b**2))
        distancia_5 = np.abs(a*mic5_cart[0] + b*mic5_cart[1] + c)/(np.sqrt(a**2+b**2))
        distancia_6 = np.abs(a*mic6_cart[0] + b*mic6_cart[1] + c)/(np.sqrt(a**2+b**2))
        
    #cálculo dos atrasos
    
    v = 34000 #velocidade do som em centímetros por segundo
    fs = 88200 #frequência de amostragem
    Ts = 1/fs
    
    tempo_1 = distancia_1/v
    tempo_2 = distancia_2/v
    tempo_3 = distancia_3/v
    tempo_4 = distancia_4/v
    tempo_5 = distancia_5/v
    tempo_6 = distancia_6/v
    
    delay1 = int(round(tempo_1/Ts))
    delay2 = int(round(tempo_2/Ts))
    delay3 = int(round(tempo_3/Ts))
    delay4 = int(round(tempo_4/Ts))
    delay5 = int(round(tempo_5/Ts))
    delay6 = int(round(tempo_6/Ts))
    
    return delay1,delay2,delay3,delay4,delay5,delay6
    
def aplica_DAS(delay1,delay2,delay3,delay4,delay5,delay6):
#     t0 = time.time()
    sinal_mult = read("entrada.wav")
    sinal_mult = np.array(sinal_mult[1])
    sinal_mult = np.transpose(sinal_mult)
    y1 = sinal_mult[0]
    y2 = sinal_mult[1]
    y3 = sinal_mult[2]
    y4 = sinal_mult[3]
    y5 = sinal_mult[4]
    y6 = sinal_mult[5]
    y7 = sinal_mult[6]
    y8 = sinal_mult[7]
    
    y1pot = np.mean(np.square(y1))
    y2pot = np.mean(np.square(y2))
    y3pot = np.mean(np.square(y3))
    y4pot = np.mean(np.square(y4))
    y5pot = np.mean(np.square(y5))
    y6pot = np.mean(np.square(y6))
    y7pot = np.mean(np.square(y7))
    y8pot = np.mean(np.square(y8))
    
    mediapot = np.mean([y1pot,y2pot,y3pot,y4pot,y5pot,y6pot,y7pot,y8pot])/2
        
    if y1pot <= mediapot and y2pot >= mediapot:
        y1a = y2
        y2a = y3
        y3a = y4
        y4a = y5
        y5a = y6
        y6a = y7
    elif y2pot <= mediapot and y3pot >= mediapot:
        y1a = y3
        y2a = y4
        y3a = y5
        y4a = y6
        y5a = y7
        y6a = y8
    elif y3pot <= mediapot and y4pot >= mediapot:
        y1a = y4
        y2a = y5
        y3a = y6
        y4a = y7
        y5a = y8
        y6a = y1
    elif y4pot <= mediapot and y5pot >= mediapot:
        y1a = y5
        y2a = y6
        y3a = y7
        y4a = y8
        y5a = y1
        y6a = y2
    elif y5pot <= mediapot and y6pot >= mediapot:
        y1a = y6
        y2a = y7
        y3a = y8
        y4a = y1
        y5a = y2
        y6a = y3
    elif y6pot <= mediapot and y7pot >= mediapot:
        y1a = y7
        y2a = y8
        y3a = y1
        y4a = y2
        y5a = y3
        y6a = y4
    elif y7pot <= mediapot and y8pot >= mediapot:
        y1a = y8
        y2a = y1
        y3a = y2
        y4a = y3
        y5a = y4
        y6a = y5
    elif y8pot <= mediapot and y1pot >= mediapot:
        y1a = y1
        y2a = y2
        y3a = y3
        y4a = y4
        y5a = y5
        y6a = y6
    else:
        y1a = np.zeros(len(y1))
        y2a = np.zeros(len(y2))
        y3a = np.zeros(len(y3))
        y4a = np.zeros(len(y4))
        y5a = np.zeros(len(y5))
        y6a = np.zeros(len(y6))
    
    # write("mic1.wav",44100,y1a)
    # write("mic2.wav",44100,y2a)
    # write("mic3.wav",44100,y3a)
    # write("mic4.wav",44100,y4a)
    # write("mic5.wav",44100,y5a)
    # write("mic6.wav",44100,y6a)
    
    t1 = time.time()
#     print("Tempo total: "+ str(t1 - t0) + " segundos")

# write("mic1.wav",44100,y1a)
# write("mic2.wav",44100,y2a)
# write("mic3.wav",44100,y3a)
# write("mic4.wav",44100,y4a)
# write("mic5.wav",44100,y5a)
# write("mic6.wav",44100,y6a)
        
    y1b = np.roll(y1a,delay1)
    y2b = np.roll(y2a,delay2)
    y3b = np.roll(y3a,delay3)
    y4b = np.roll(y4a,delay4)
    y5b = np.roll(y5a,delay5)
    y6b = np.roll(y6a,delay6)
    
    
    yfinal = (y1b + y2b + y3b + y4b + y5b + y6b)/6
    write("saida.wav", 44100 ,yfinal.astype(np.int16))
    

delay1,delay2,delay3,delay4,delay5,delay6 = direcaochegada(5*np.pi/3)
    
    
    
    
while True:
    grava = threading.Thread(target=grava_mic,args=())
    play = threading.Thread(target=play_mic,args=())
    
    
    grava.start()
    play.start()
    
    grava.join()
    play.join()
    aplica_DAS(delay1,delay2,delay3,delay4,delay5,delay6)
    
    
    