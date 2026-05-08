import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from scipy.fft import fft, fftshift, fftfreq
from scipy.signal import spectrogram





def plot_time(iq, fs: int = 200e3,
              title: Optional[str] = None,
              n_samples: Optional[int] = None,
              I_ax=None,
              Q_ax=None):
    '''
    Plots the time domain of the signal as two plots I and Q.
    
    n_samples: Number of samples to be plotted
    I_ax: the I axis, will be created if not passed in the argument
    Q_ax: Q axis
    '''

    I = iq.real
    Q = iq.imag

    N = len(I)
    t = np.arange(N) / fs * 1e6
    
    if I_ax is None or Q_ax is None:
        _, (I_ax, Q_ax) = plt.subplots(2,1, figsize=(12,5), sharex=True)


    I_ax.set_ylabel('I')
    Q_ax.set_ylabel('Q')
    Q_ax.set_xlabel(r'Time ($\mu$s)')

    if title is not None:
        I_ax.set_title(title)

    I_ax.plot(t, I)
    Q_ax.plot(t, Q)

    return I_ax, Q_ax

def plot_fft(iq, fs: int=200e3,
             title: Optional[str] = None,
             ax=None):
    I = iq.real
    Q = iq.imag

    signal = I + 1j * Q

    IQ_fft = fftshift(fft(signal))
    power_db = 20 * np.log10(np.abs(IQ_fft) + 1e-10)

    n = I.size
    f = fftshift(fftfreq(n, d=1/fs))

    if ax is None:
        _, ax = plt.subplots(1,1)

    ax.plot(f/1e3, power_db)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Power (dB)')

    if title is not None:
        ax.set_title(title)

    return ax
    


def plot_spectrogram(iq, fs: int=200e3,
                     title: Optional[str]=None,
                     ax=None):
    I = iq.real
    Q = iq.imag

    IQ = I + 1j *  Q

    f, t, Sxx = spectrogram(IQ, fs=fs, nperseg=64, return_onesided=False)
    Sxx_db    = 10 * np.log10(np.abs(Sxx) + 1e-10)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    mesh = ax.pcolormesh(t * 1e3, f / 1e3, Sxx_db,
                         shading='gouraud', cmap='inferno')
    plt.colorbar(mesh, ax=ax, label='Power (dB)')
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Frequency (kHz)', fontsize=11)
    ax.set_ylim(-100, 0)
    ax.set_title(title or 'Spectrogram', fontsize=13)

    return ax
    

def plot_constellations(iq, title: Optional[str] = None,
                        ax=None):

    if ax is None:
        _, ax = plt.subplots(1,1)
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.axhline(0)
    ax.axvline(0)


    ax.scatter(iq.real, iq.imag)
    
    return ax


def make_fig(signals: dict, 
             plot_fn,
             title: Optional[str] = None,
             **plot_kwargs):
    '''
    Creates a figure showing the plots for all passed signal types for comparison, 
    can be used to create stand alone plots.

    signals: A dictionary with each signal type ("BPSK", "QPSK", etc) as the key,
             and the generated iq as  the value.
    
    plot_fn: The type of plot the user wishes to create (plot_time, plot_fft, etc.

    title: Title for the overall figure
    '''
    n = len(signals)
    #plot_time needs a separate figure creation to deal with the separate I and Q axes.
    if plot_fn == plot_time:
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 6))
        for i, (name, iq) in enumerate(signals.items()):
            plot_time(iq, I_ax=axes[0, i], Q_ax=axes[1, i],
                             title=name, **plot_kwargs)
    #figure creation for all other plot types
    else:
        fig,axes = plt.subplots(1, n, figsize=(5*n, 4))
        if n == 1:
            axes = [axes]
        for ax, (name, iq) in zip(axes, signals.items()):
            plot_fn(iq, ax=ax, title=name, **plot_kwargs)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


