import io
import time
import torch
import numpy as np
from pydub import AudioSegment


def tokenise(audio_np_array: np.ndarray) -> torch.Tensor:
    """
    Function to tokenise an audio file represented as a NumPy array.

    Args:
    - audio_np_array (np.ndarray): The audio file as a NumPy array.

    Returns:
    - torch.Tensor: A random 1D tensor with dtype int16 and variable length in range (20, 1000).
    """

    # Check if the input is a NumPy array
    if not isinstance(audio_np_array, np.ndarray):
        raise ValueError("Input should be a NumPy array")

    # Time delay to simulate model inference
    time.sleep(0.15)

    tensor_length = np.random.randint(20, 1001)  # 1001 is exclusive
    return torch.randint(low=-32768, high=32767, size=(tensor_length,), dtype=torch.int16)


def convert_flac_to_wav(flac_data: np.array):
    """
    Convert FLAC data to WAV using pydub
    Args:
    - flac_data (np.ndarray): The flac audio file as a NumPy array.
    """
    audio = AudioSegment.from_file(io.BytesIO(flac_data), format='flac')
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    return audio.export(format='wav').read()