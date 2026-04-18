import numpy as np 
import matplotlib.pyplot as plt
from typing import Optional


def generate_bpsk(n_symbols: int, sps: int = 8,
                  snr_db: Optional[float] = None) -> np.ndarray:
    """
    Binary Phase Shift Keying.
    
    Maps bits to {+1, -1} on real axis. Q channel is zero.
    Pulse shape: rectangular (for simplicity)
    
    n_symbols : number of BPSK symbols
    sps       : samples per symbol
    snr_db    : if provided, add AWGN at this SNR
    returns   : complex IQ array, shape (n_symbols * sps,)
    """
    bits    = np.random.randint(0, 2, n_symbols)
    symbols = 2 * bits - 1                        # {0,1} → {-1, +1}
    signal  = np.repeat(symbols, sps).astype(complex)
    return signal


def generate_qpsk(n_symbols: int, sps: int = 8, 
                    snr_db: Optional[float] = None) -> np.ndarray:
    """
    Quadrature Phase Shift Keying

    Maps symbols to 45, 135, 225, 315 degrees.
    Symbols carry two bits each.

    returns complex IQ array, shape (n_symbols * sps)
    """
    bits = np.random.randint(0, 2, (n_symbols, 2)) #2D Matrix
    I = (2 * bits[:,0] - 1) / np.sqrt(2)
    Q = (2 * bits[:,1] - 1) / np.sqrt(2)
    symbols = I + 1j * Q
    signal = np.repeat(symbols, sps)
    return signal

def generate_8psk(n_symbols: int, sps: int = 8,
                    snr_db: Optional[float] = None) -> np.ndarray:
    """
    8 Phase Shift Keying

    Maps symbols to 8 points spaced equally by 45 degrees.
    Each symbol carries 3 bits

    returns complex IQ array, shape (n_symbols * sps)
    """
    indices = np.random.randint(0, 8, n_symbols)
    angles = 2*np.pi * indices / 8
    symbols = np.exp(1j*angles)
    signal = np.repeat(symbols, sps)
    return signal

def generate_qam(n_symbols: int, order: int = 16, sps: int = 8, 
                    snr_db: Optional[float] = None) -> np.ndarray:
    """
    Quadrature Amplitude Modulation

    order: Constellation size - must be perfect square (4,16,64,256)
    returns square QAM Grid normalized to unit average power.
    """
    assert order in (4,16,64,256), "Order must be 4, 16, 64, or 256"
    m = int(np.sqrt(order))
    level = np.arange(-(m-1), m, 2)
    I = np.random.choice(level, n_symbols)
    Q = np.random.choice(level, n_symbols)
    symbols = I + 1j*Q
    #normalized to unit average power
    symbols /= np.sqrt(np.mean(np.abs(symbols) ** 2))
    signal = np.repeat(symbols, sps)
    return signal

def generate_fm(n_samples: int, fs: float = 48000.0, 
                    fc: float = 0.0, kf: float = 5000.0,
                    snr_db: Optional[float] = None) -> np.ndarray:
    """
    Frequency Modulation

    fs: Sample Rate (Hz)
    fc: Carrier Frequency Offset (Hz) - 0 = baseband
    kf: Frequency Deviation (Hz)

    Signal is the sum of three random sinusoids simulating audio content
    FM: s(t) = exp(j*2pi*kf/fs*cumsum(message))
    """
    t = np.arange(n_samples) / fs
    #3 Random tones between 300Hz and 3400Hz
    freqs = np.random.uniform(300, 3400, 3)
    message = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    message /= np.max(np.abs(message)) #Normalized to [-1, 1]
    phase = 2 * np.pi * kf * np.cumsum(message) / fs
    signal = np.exp(1j * (2 * np.pi * fc * t + phase))
    return signal

def generate_fsk(n_symbols: int, fs: float = 48000.0,
                 fdev: float = 5000.0,
                 sps: int = 8,
                 snr_db: Optional[float] = None) -> np.ndarray:
    """
    Binary Frequency Shift Keying.
    
    Bit 0 → carrier at -fdev Hz
    Bit 1 → carrier at +fdev Hz
    
    fdev : frequency deviation from center (Hz)
    """
    bits    = np.random.randint(0, 2, n_symbols)
    freqs   = np.where(bits == 1, fdev, -fdev)     # one freq per symbol
    # Expand to samples and integrate phase for continuity
    freq_samples = np.repeat(freqs, sps).astype(float)
    phase   = 2 * np.pi * np.cumsum(freq_samples) / fs
    signal  = np.exp(1j * phase)
    return signal



###Impairments###

def freq_offset(signal: np.ndarray, fs: float, offset_hz: float) -> np.ndarray:
    '''Adds frequency off set to input signal'''
    t = np.arange(len(signal)) / fs
    phasor = np.exp(1j * 2 * np.pi * offset_hz * t)
    return signal * phasor

def phase_noise(signal: np.ndarray, phase_std_rad: float = 0.05) -> np.ndarray:
    '''Adds random phase perturbation for each sample'''
    p_noise = np.cumsum(np.random.randn(len(signal)) * phase_std_rad)
    return signal * np.exp(1j * p_noise)


def add_iq_imbalance(signal: np.ndarray, amplitude_imbalance_db: float = 1.0, 
                 phase_imbalance_deg: float = 2.0) -> np.ndarray:
    """
    Adds IQ imbalance to signal.
    alpha is the amplitude imbalance
    phi is the phase imbalance
    """
    alpha = 10 ** (amplitude_imbalance_db / 20)
    phi = np.deg2rad(phase_imbalance_deg)
    I_out = signal.real * alpha
    Q_out = alpha * np.sin(phi) * signal.real + np.cos(phi) * signal.imag
    return I_out + 1j * Q_out

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add complex AWGN to a signal at a specified SNR in dB.
    
    signal : complex IQ array, shape (N,)
    snr_db : desired signal-to-noise ratio in dB
    returns: noisy complex signal, same shape
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear   = 10 ** (snr_db / 10)
    noise_power  = signal_power / snr_linear
    # Complex noise: real and imag each get half the noise power
    noise = (np.sqrt(noise_power / 2) *
             (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))))
    return signal + noise

def add_impairments(signal: np.ndarray, fs: float, snr_db: float = 20.0,
                    offset_hz: float = 0.0, 
                    phase_noise_std: float = 0.0,
                    iq_imbalance: bool = False) -> np.ndarray:
    '''Adds any or all impairments in one call'''
    if offset_hz != 0:
        signal = freq_offset(signal, fs, offset_hz)
    if phase_noise_std > 0:
        signal = phase_noise(signal, phase_noise_std)
    if iq_imbalance:
        signal = add_iq_imbalance(signal)
    signal = add_awgn(signal, snr_db)
    return signal

