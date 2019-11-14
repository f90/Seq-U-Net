import h5py
import torch
import librosa
from torch.utils.data import Dataset
import os
import numpy as np
from sortedcontainers import SortedList
import utils

def load(path, sr=22050, mono=True, mode="numpy"):
    y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast')

    if mono or len(y.shape) == 1:
        # Expand channel dimension
        y = y[np.newaxis, :]

    if mode == "pytorch":
        y = torch.tensor(y)

    return y, curr_sr

class AudioData(object):
    def __init__(self, hdf_path, sr, channels, parallel_cores=1):
        # Check if HDF file exists already
        if os.path.exists(hdf_path):
            # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
            with h5py.File(hdf_path, "r") as f:
                if f.attrs["sr"] != sr or f.attrs["channels"] != channels:
                    raise ValueError("Tried to load existing HDF file, but sampling rate and channel are not as expected. Did you load an out-dated HDF file?")
        else:
            # Create HDF file
            with h5py.File(hdf_path, "w") as f:
                f.attrs["sr"] = sr
                f.attrs["channels"] = channels

        self.path = hdf_path
        self.sr = sr
        self.channels = channels
        self.parallel_cores = parallel_cores

    def add(self, examples):
        if isinstance(examples, str):
            examples = [examples]

        if not all([isinstance(ex, str) for ex in examples]):
            raise SyntaxError("You need to feed one more str objects!")

        for example in examples:
            # In this case, read in audio and convert to target sampling rate
            audio, _ = load(example, sr=self.sr, mono=True, mode="pytorch")

            # Mu-law
            audio = utils.mu_law_encoding(audio, qc=256).numpy().astype(np.uint8)

            # Add to HDF5 file
            with h5py.File(self.path, "a") as f:
                curr_len = str(len(self))
                grp = f.create_group(curr_len)

                grp.create_dataset("audio", shape=audio.shape, dtype=audio.dtype, data=audio)
                grp.attrs["length"] = audio.shape[1]

            print("Added audio file to dataset")

    def __len__(self):
        with h5py.File(self.path, "r") as f:
            return len(f)

    def is_empty(self):
        return (len(self) == 0)

class AudioDataset(Dataset):
    def __init__(self, data:AudioData, input_size=None, context_front=0, context_back=0, hop_size=100, random_hops=False, audio_transform=None, in_memory=False):
        '''

        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. This chops off ends of audio signals if pad_end is False! If True, randomly sample a position from the audio
        '''

        self.hdf_dataset = None
        self.hdf_path = data.path
        self.channels = data.channels

        self.input_size = input_size
        self.hop_size = hop_size
        self.context_front = context_front
        self.context_back = context_back
        self.random_hops = random_hops
        self.audio_transform = audio_transform
        self.in_memory = in_memory

        # Go through HDF and collect lengths of all audio files
        if input_size != None:
            with h5py.File(self.hdf_path, "r") as f:
                lengths = [f[str(song_idx)].attrs["length"] for song_idx in range(len(f))]

                # Subtract input_size from lengths and divide by hop size to determine number of starting positions
                lengths = [((l - input_size + 1) // hop_size) + 1 for l in lengths]

            self.start_pos = SortedList(np.cumsum(lengths))
            self.length = self.start_pos[-1]
        else:
            raise NotImplementedError

    def __getitem__(self, index):

        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_path, 'r', driver=driver)

        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            index = index - self.start_pos[audio_idx-1]

        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]

        if self.random_hops:
            # Take random position from song
            start_pos = np.random.randint(0, audio_length - self.input_size + 1)
        else:
            # Map item index to sample position within song
            start_pos = index * self.hop_size
        end_pos = start_pos + self.input_size

        if start_pos >= audio_length:
            print("WRONG!")

        # Check front padding
        start_pos -= self.context_front
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos += self.context_back
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read and return
        audio = self.hdf_dataset[str(audio_idx)]["audio"][:,start_pos:end_pos].astype(np.int64)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0,0), (pad_front, pad_back)], mode="constant", constant_values=128)

        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        return audio

    def __len__(self):
        return self.length