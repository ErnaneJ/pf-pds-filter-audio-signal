import numpy as np
from matplotlib import pyplot as plt

from scipy.fft import fft, fftfreq, ifft
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.signal import welch

# ==================================| A | ================================== #
# Plote o sinal no domínio do tempo.

# Carrega o arquivo
samplerate, data = wavfile.read('./audios/piano.wav')
data = data[:len(data)//2]

# Carrega o arquivo em dois canais (audio estereo)
print(f"numero de canais = {data.shape[1]}")

# Tempo total = numero de amostras / fs
length = data.shape[0] / samplerate
print(f"duracao = {length}s")
print(f"numero de amostras = {data.shape[0]}")
print(f"frequencia de amostragem = {samplerate}Hz")

# Interpola para determinar eixo do tempo
time = np.linspace(0., length, data.shape[0])

nsampples=data.shape[0]

# Seleciona os canais esquerdo e direito
data_l = data[:, 0]
data_r = data[:, 1]

# Plota os canais esquerdo e direito
plt.figure(1)
plt.plot(time, data_l)
plt.title("Canal esquerdo")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.show()

plt.figure(2)
plt.plot(time, data_r)
plt.title("Canal direito")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.show()

# ==================================| B | ================================== #
# Usando a transformada rápida de Fourier (FFT), de alguma biblioteca de Python, plote o espectro de
# frequências do sinal para as primeiras N amostras. Use um valor de N adequado. Escalone o eixo de
# frequências das N amostras da FFT de forma adequada.

N = 2**18 # 2**18 = 262144 o resto fica com zero "menor maior possivel".

x = data_l
X = fft(x, N)
X_f = fftfreq(N, 1/samplerate)
plt.figure(3)
plt.plot(X_f, abs(X))
plt.xlim(-1500, 1500)
plt.title("DFT de piano.wav canal esquerdo")
plt.xlabel("Frequência [kHz]")
plt.ylabel("Amplitude")
plt.show()

x = data_r
X = fft(x, N)
X_f = fftfreq(N, 1/samplerate)
plt.figure(4)
plt.plot(X_f, abs(X))
plt.xlim(-1500, 1500)
plt.title("DFT de piano.wav canal direito")
plt.xlabel("Frequência [kHz]")
plt.ylabel("Amplitude")
plt.show()

# ==================================| C | ================================== #
# Utilizando os conceitos de mudança de taxa de amostragem vistos em aula, interpole o sinal por um
# fator L = 3;

data_r3 = [(data_r[i//3] if i % 3 == 0 else 0) for i in range(3*len(data_r) - 2)]
data_l3 = [(data_l[i//3] if i % 3 == 0 else 0) for i in range(3*len(data_l) - 2)]

D = fft(data_r, N)
D_f = fftfreq(N, 1/samplerate)

plt.figure(5)
plt.title("DFT do canal direito")
plt.xlabel("Frequência [kHz]")
plt.ylabel("Amplitude")
#plt.xlim(-1500, 1500)
plt.plot(D_f, abs(D))
plt.show()

N3 = 2**22

D3 = fft(data_r3, N3)
D3_f = fftfreq(N3, 1/(3*samplerate))

plt.figure(6)
plt.title("DFT do canal direito expandido por L = 3")
plt.xlabel("Frequência [kHz]")
plt.ylabel("Amplitude")
plt.xlim(-1500, 1500)
plt.plot(D3_f, abs(D3))
plt.show()

# ==================================| C.1 | ================================== #
# Fazendo uso da biblioteca pyFDA, projete um filtro passa-baixas digital, 
# com resposta ao impulso finita (FIR), adequado para a interpolação por L = 3;

h = 3*np.genfromtxt('./coefficients/coeffk.csv', delimiter=',') # coeficientes x3 para compensar a interpolação L=3

plt.figure(7)
plt.stem(h)
plt.show()

# ==================================| C.2 | ================================== #
# Com os coeficientes do filtro projetado, filtre o sinal. Implemente a filtragem 
# com a operação de convolução no domínio da frequência, fazendo uso de funções FFT e 
# IFFT (utilize o método de sobreposição e soma ou sobreposição e armazenamento);

x = data_r3
P = len(h)
L = 998
N = P + L - 1
xLen = len(x)
NT = P + xLen - 1
iter = xLen//L if xLen % L == 0 else xLen//L + 1

H = fft(h, N)

y = np.zeros(NT)

for i in range(iter):
    X_r = fft(x[L*i:min(L*(i + 1), xLen)], N)
    Y_r = X_r*H
    y_r = ifft(Y_r, N)
    y[i*L:i*L + N] = y[i*L:i*L + N] + y_r[:len(y[i*L:i*L + N])]

data_r3_inter = y

x = data_l3
P = len(h)
L = 998
N = P + L - 1
xLen = len(x)
NT = P + xLen - 1
iter = xLen//L if xLen % L == 0 else xLen//L + 1

H = fft(h, N)

y = np.zeros(NT)

for i in range(iter):
    X_r = fft(x[L*i:min(L*(i + 1), xLen)], N)
    Y_r = X_r*H
    y_r = ifft(Y_r, N)
    y[i*L:i*L + N] = y[i*L:i*L + N] + y_r[:len(y[i*L:i*L + N])]

data_l3_inter = y

# ==================================| D | ================================== #
# Plote os conteúdos temporais e espectrais do sinal original, do sinal após a expansão e 
# após a interpolação. (O conteúdo espectral pode ser plotado com funções que implementem a 
# FFT ou com a função welch.)

audio = np.array([data_l3_inter, data_r3_inter]).T
scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
filename = './audios/piano_interp_L3.wav'
write(filename, samplerate*3, scaled)

plt.figure(7, (14,4), dpi=160)
data_lx = np.linspace(0,len(data_l[:30])*1/samplerate, len(data_l[:30]))
plt.plot(data_lx ,data_l[:30], color="red", label="canal esq")
data_l3x = np.linspace(0,len(data_l3[:90])*1/(2.93*samplerate), len(data_l3[:90]))
plt.stem(data_l3x ,data_l3[:90], label="canal esq expandido", basefmt="black")
plt.legend()
# plt.ylim(-0.1,2.5)
plt.xlabel("tempo [s]")
plt.show()

plt.figure(7, (14,4), dpi=160)
plt.stem(data_lx, data_l[:30], label="canal esq", basefmt="black")
data_l3_interx = np.linspace(0,len(data_l3[:90])*1/(2.93*samplerate), len(data_l3[:90]))
plt.plot(data_l3_interx, data_l3_inter[13:103], label="canal esq interpolado", color="red")
plt.legend()
# plt.ylim(-0.3,3)
plt.xlabel("tempo [s]")
plt.show()

plt.figure(9, (14,4), dpi=160)
data_rx = np.linspace(0,len(data_r[:30])*1/samplerate, len(data_r[:30]))
plt.plot(data_rx, data_r[:30], color="red", label="canal dir")
data_r3x = np.linspace(0,len(data_r3[:90])*1/(3.0*samplerate), len(data_r3[:90]))
plt.stem(data_r3x, data_r3[:90], label="canal dir expandido", basefmt="black")
plt.legend()
# plt.ylim(-1.1,2.5)
plt.xlabel("tempo [s]")
plt.show()

plt.figure(10, (14,4), dpi=160)
plt.stem(data_rx, data_r[:30], label="canal dir", basefmt="black")
data_r3_interx = np.linspace(0,len(data_r3[:90])*1/(3.0*samplerate), len(data_r3[:90]))
plt.plot(data_r3_interx, data_r3_inter[13:103], label="canal dir interpolado", color="red")
plt.legend()
# plt.ylim(-1.1,3)
plt.xlabel("tempo [s]")
plt.show()

plt.figure(11, (14,4), dpi=160)
x  = data_l
fs = 2
f, Pxx_spec = welch(x, fs, 'flattop', 512, scaling='spectrum')
plt.title("espectro canal esq")
plt.ylabel("espectro")
plt.xlabel("frequência [rad/($\pi$)]")
plt.semilogy(f, Pxx_spec)
plt.show()

plt.figure(12, (14,4), dpi=160)
x  = data_l3
fs = 2
f, Pxx_spec = welch(x, fs, 'flattop', 512, scaling='spectrum')
plt.title("espectro canal esq expandido")
plt.ylabel("espectro")
plt.xlabel("frequência [rad/($\pi$)]")
plt.semilogy(f, Pxx_spec)
plt.show()

plt.figure(13, (14,4), dpi=160)
x  = data_l3_inter
fs = 2
f, Pxx_spec = welch(x, fs, 'flattop', 512, scaling='spectrum')
plt.title("espectro canal esq interpolado")
plt.ylabel("espectro")
plt.xlabel("frequência [rad/($\pi$)]")
plt.semilogy(f, Pxx_spec)
plt.show()

plt.figure(14, (14,4), dpi=160)
x  = data_r
fs = 2
f, Pxx_spec = welch(x, fs, 'flattop', 512, scaling='spectrum')
plt.title("espectro canal dir")
plt.ylabel("espectro")
plt.xlabel("frequência [rad/($\pi$)]")
plt.semilogy(f, Pxx_spec)
plt.show()

plt.figure(15, (14,4), dpi=160)
x  = data_r3
fs = 2
f, Pxx_spec = welch(x, fs, 'flattop', 512, scaling='spectrum')
plt.title("espectro canal dir expandido")
plt.ylabel("espectro")
plt.xlabel("frequência [rad/($\pi$)]")
plt.semilogy(f, Pxx_spec)
plt.show()

plt.figure(16, (14,4), dpi=160)
x  = data_r3_inter
fs = 2
f, Pxx_spec = welch(x, fs, 'flattop', 512, scaling='spectrum')
plt.title("espectro canal dir interpolado")
plt.ylabel("espectro")
plt.xlabel("frequência [rad/($\pi$)]")
plt.semilogy(f, Pxx_spec)
plt.show()