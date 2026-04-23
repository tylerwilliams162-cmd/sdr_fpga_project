import matplotlib.pyplot as plt
import numpy as np
from datasets import SyntheticData
from typing import Optional
from scipy.fft import fft, fftshift, fftfreq
from scipy.signal import spectrogram
from synth import (generate_bpsk, generate_qpsk, generate_8psk,
                   generate_qam, generate_fm, generate_fsk)





def plot_time(iq, fs: int = 200e3,
              title: Optional[str] = None):

    I = iq[0]
    Q = iq[1]

    N = len(I)
    t = np.arange(N) / fs * 1e6
    
    fig, (I_ax, Q_ax) = plt.subplots(2,1, figsize=(12,5))
    I_ax.set_ylabel('I')
    Q_ax.set_ylabel('Q')
    Q_ax.set_xlabel(r'Time ($\mu$s)')

    if title is not None:
        I_ax.set_title(title)

    I_ax.plot(t, I)
    Q_ax.plot(t, Q)

    plt.tight_layout()
    plt.show()

def plot_fft(iq, fs: int=200e3,
             title: Optional[str] = None):
    I = iq[0]
    Q = iq[1]

    signal = I + 1j * Q

    IQ_fft = fftshift(fft(signal))
    power_db = 20 * np.log10(np.abs(IQ_fft) + 1e-10)

    n = I.size
    f = fftshift(fftfreq(n, d=1/fs))

    fig, ax = plt.subplots(1,1)
    ax.plot(f/1e3, power_db)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Power (dB)')

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_spectrogram(iq, fs: int=200e3,
                     title: Optional[str]=None):
    I = iq[0]
    Q = iq[1]

    IQ = I + 1j *  Q

    f, t, Sxx = spectrogram(IQ, fs=fs, nperseg=64)
    Sxx_db    = 10 * np.log10(np.abs(Sxx) + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(t * 1e3, f / 1e3, Sxx_db,
                         shading='gouraud', cmap='inferno')
    plt.colorbar(mesh, ax=ax, label='Power (dB)')
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Frequency (kHz)', fontsize=11)
    ax.set_ylim(-100, 0)
    ax.set_title(title or 'Spectrogram', fontsize=13)
    plt.tight_layout()
    plt.show()
    

def plot_constellations(iq, title: Optional[str] = None):

    
    fig, ax = plt.subplots(1,1)
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.axhline(0)
    ax.axvline(0)


    ax.scatter(iq[0], iq[1])
    plt.show()



sig = generate_qpsk(32, snr_db=20.0)
iq = ([sig.real, sig.imag])


#plot_fft(iq, title="Synthetic BPSK PSD")
#plot_time(iq, title='Synthetic IQ Time Domain')
#plot_constellations(iq)
plot_spectrogram(iq)