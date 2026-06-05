import numpy as np
from synth import (generate_bpsk, generate_8psk, 
                   generate_fm, generate_fsk, 
                   generate_qam, generate_qpsk)
from datasets import SyntheticData, MODULATION_CLASSES, CLASS_TO_INDEX
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def moment(s, p, q):
    return np.mean((s ** p) * (np.conj(s) ** q), axis=1)


def features(iq: np.ndarray) -> np.ndarray:
    '''
    Returns an array of features to be used for classification. Features include:

    Instantaneous Features:
        -Amplitude Standard Deviation
        -Phase Standard Deviation
        -Frequency Mean
        -Frequency Standard Deviation
        -Maximum of Power Spectral Density
        -Kurtosis
        -Skewness
    
    Higher Order Cumulants:
        -C4,0
        -C4,2
        -C6,3    

    
    '''
    I = iq[:, 0, :]
    Q = iq[:, 1, :]

    amp = np.sqrt(I**2 + Q**2)
    amp_mean = np.mean(amp, axis=1, keepdims=True)
    amp_scaled = amp / amp_mean
    amp_std = np.std(amp_scaled, axis=1)


    raw_phase = np.arctan2(Q, I)
    phase_diff = np.diff(raw_phase, axis=1)
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    last_col = phase_diff[:, -1:]
    phase_diff = np.hstack((phase_diff, last_col))
    phase_std = np.std(phase_diff, axis=1)


    freq = (1 / (2 * np.pi)) * phase_diff
    freq_std = np.std(freq, axis=1)
    freq_mean = np.mean(freq, axis=1)

    amp_kurt = kurtosis(amp_scaled, axis=1)
    amp_skew = skew(amp_scaled, axis=1)

    s = I + 1j*Q

    #Moments
    M20 = moment(s, 2, 0)
    M21 = moment(s, 2, 1)
    M22 = moment(s, 2, 2)
    M40 = moment(s, 4, 0)
    M41 = moment(s, 4, 1)
    M42 = moment(s, 4, 2)
    M60 = moment(s, 6, 0)
    M63 = moment(s, 6, 3)

    #Cumulants
    C40 = np.abs(M40 - 3 * (M20**2))
    C42 = np.abs(M42 - (np.abs(M20)**2) - 2 * (M22**2))
    C60 = np.abs(M60 - 15*M40*M20 + 30*(M20**3))
    C63 = np.abs(M63 - 6*M42*M21 - 9*M41*M22 + 18*(M22**2)*M21)

    _, psd = welch(s, axis=1, return_onesided=False)

    max_psd = np.max(psd, axis=1)

    return np.column_stack((amp_std, phase_std, freq_std, 
                            freq_mean, amp_kurt, amp_skew,
                            C40, C42, C60, C63, max_psd))


