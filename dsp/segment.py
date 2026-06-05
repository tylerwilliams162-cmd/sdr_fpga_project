import numpy as np

def segment_frames(iq, 
                   frame_len: int = 256,
                   overlap: int = 0,
                   )->np.ndarray:
    

    step = frame_len - overlap
    n_frames = (len(iq) - frame_len) // step + 1

    frames = np.zeros((n_frames, 2, frame_len), dtype=np.float32)

    for i in range(n_frames):
        start = i * step
        chunk = iq[start : start + frame_len]
        frames[i, 0] = chunk.real
        frames[i, 1] = chunk.imag

    return frames
    

def normalize_frames(frames: np.ndarray) -> np.ndarray:
    
    mean = frames.mean(axis=(1, 2), keepdims=True)  # (n_frames, 1, 1)
    std  = frames.std(axis=(1, 2),  keepdims=True)  # (n_frames, 1, 1)

    # Avoid division by zero
    std  = np.where(std < 1e-8, 1e-8, std)

    return (frames - mean) / std