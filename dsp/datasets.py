from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
from synth import (generate_bpsk, generate_qpsk, generate_8psk,
                      generate_qam, generate_fm, generate_fsk, 
                      add_impairments)


MODULATION_CLASSES = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "FM", "FSK"]
N_CLASSES = len(MODULATION_CLASSES)
CLASS_TO_INDEX = {c: i for i, c in enumerate(MODULATION_CLASSES)}


class DataSource(ABC):

    @abstractmethod
    def get_frame(self,
                  n_frames: int, 
                  frame_len: int = 256,
                  snr_db: Optional[float] = 20.0
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @property
    def class_names(self):
        return MODULATION_CLASSES
    

class SyntheticData(DataSource):

    def __init__(self, fs: int = 200e3,
                 sps: int = 8,
                 add_impairments: bool = False,
                 snr: float=20.0):
        self.fs = fs
        self.sps = sps
        self.add_impairments = add_impairments

        self.generators = {
                            "BPSK": lambda n, snr: generate_bpsk(int(n // self.sps), sps=self.sps, snr_db=snr),
                            "QPSK": lambda n, snr: generate_qpsk(int(n // self.sps), sps=self.sps, snr_db=snr),
                            "8PSK": lambda n, snr: generate_8psk(int(n // self.sps), sps=self.sps, snr_db=snr),
                            "QAM16": lambda n, snr: generate_qam(int(n // self.sps), order=16, sps=self.sps, snr_db=snr),
                            "QAM64": lambda n, snr: generate_qam(int(n // self.sps), order=64, sps=self.sps, snr_db=snr),

                             # FM is sample‑based, not symbol‑based
                            "FM": lambda n, snr: generate_fm(n, fs=self.fs, fc=0.0, kf=5000.0, snr_db=snr),

                            # FSK needs keyword args to avoid argument misalignment
                            "FSK": lambda n, snr: generate_fsk(int(n // self.sps), fs=self.fs, sps=self.sps, snr_db=snr),
}

        
    def get_frame(self,
                  n_frames: int, 
                  frame_len: int = 256,
                  snr_db: Optional[float] = 20.0
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates synthetic signal frames and outputs a tuple of the format (iq, labels, snrs)
        iq is the signal frame, labels is one of the MODULATION_CLASSES (BPSK, QPSK, etc.), snrs 
        is the signal to noise ratio for each frame.
        """
        iq = np.zeros((n_frames, 2, frame_len), dtype=np.float32)
        labels = np.zeros(n_frames, dtype=np.int64)
        snrs = np.full(n_frames, snr_db, dtype=np.float32)


        for i in range(n_frames):
            #randomly pick a modulation class
            class_name = np.random.choice(MODULATION_CLASSES)
            label = CLASS_TO_INDEX[class_name]

            snr = (snr_db + np.random.uniform(-2, 2)
                   if snr_db is not None else 20.0)
                
            sig = self.generators[class_name](frame_len, snr)


            if self.add_impairments:
                sig = add_impairments(
                        sig, self.fs,
                    snr_db=100,                # noise already added
                    freq_offset_hz=np.random.uniform(-500, 500),
                    phase_noise_std=0.02,
                    iq_imbalance=(np.random.rand() > 0.7)
                )

            sig = sig[:frame_len]
            if len(sig) < frame_len:
                sig = np.pad(sig, (0, frame_len - len(sig)))


            iq[i, 0, :] = sig.real
            iq[i, 1, :] = sig.imag
            labels[i]   = label
            snrs[i]      = snr

        return iq, labels, snrs
        