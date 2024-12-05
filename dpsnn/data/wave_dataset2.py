import h5py
import csv
import os
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
import glob


def worker_init_fn(worker_id): # This is apparently needed to ensure workers have different random seeds and draw different examples!
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class ContextSepDataset(Dataset):
    def __init__(self, hdf_file, frame_dur, sr, channels, start_context_dur, end_context_dur, random_hops=False):
        '''
        Stream version of SeparationDataset. For each utterance, it ignores the first context 
        and end context to avoid padding, and it ignores the last segment to aovid padding.
        

        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(ContextSepDataset, self).__init__()

        self.random_hops = random_hops
        self.sr = sr
        self.hdf_file = hdf_file
        self.channels = channels
        self.shapes = None
        self.hdf_dataset = None

        self.output_size = int(frame_dur * sr)
        self.start_context_size = int(start_context_dur * sr)
        self.end_context_size = int(end_context_dur * sr)
        self.input_size = self.output_size + self.start_context_size + self.end_context_size
        self.shapes = self._set_shapes()

    def _set_shapes(self):
        shapes = {"input_size" : self.input_size,
                  "output_size" : self.output_size,
                  "start_context_size" : self.start_context_size,
                  "end_context_size" : self.end_context_size}
        return shapes

    def get_shapes(self):
        return self.shapes

    def __getitem__(self, index):
        if self.hdf_dataset is None:
            self.hdf_dataset = h5py.File(self.hdf_file, 'r')

        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            # rel_indx is the frame index within a song
            rel_index = index - self.start_pos[audio_idx - 1]
        else:
            # audio_idx == 0
            rel_index = index

        name = self.hdf_dataset[str(audio_idx)].attrs["ID"]
        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]
        clean_length = self.hdf_dataset[str(audio_idx)].attrs["clean_length"]
        assert(audio_length == clean_length)

        audio = self.hdf_dataset[str(audio_idx)]["noisy"][()].astype(np.float32).squeeze()
        clean = self.hdf_dataset[str(audio_idx)]["clean"][()].astype(np.float32).squeeze()
        if audio_length < self.input_size:
            pad_back = self.input_size - audio_length
            if pad_back > audio_length:
                pad_times = pad_back // audio_length
                padded = np.tile(audio, pad_times)
                audio = np.append(audio, [np.tile(audio, pad_times)])
                clean = np.append(clean, [np.tile(clean, pad_times)])
                
                pad_back = pad_back % audio_length
            audio = np.append(audio, [audio[-pad_back:]])
            clean = np.append(clean, [clean[-pad_back:]])
            assert(len(audio) == self.input_size, f"{len(audio)}, {self.input_size}")

        # Determine position where to start clean
        if self.random_hops:
            start_pos = np.random.randint(0, max(audio_length-self.input_size, 1))
        else:
            # Map item index to sample position within song
            start_pos = rel_index * self.output_size

        # READ INPUTS

        # Check back padding
        end_pos = start_pos + self.input_size
        assert(end_pos < len(audio), f"start_pos: {start_pos}, end_pos: {end_pos}, audio_length: {audio_length}")

        # Read and return
        noisy_audio = audio[start_pos:end_pos]
        clean_audio = clean[start_pos:end_pos]

        # print(f"noisy_audio shape: {noisy_audio.shape}")
        return [name, noisy_audio, len(noisy_audio)], [name, clean_audio, len(clean_audio)]

    def __len__(self):
        if self.hdf_dataset is None:
            with h5py.File(self.hdf_file, "r") as f:
                if f.attrs["sr"] != self.sr or \
                        f.attrs["channels"] != self.channels:
                    raise ValueError(
                        "Tried to load existing HDF file, but sampling rate and channel are not as expected. "
                        "Did you load an out-dated HDF file?")
                
                # Ignore end context to avoid padding
                lengths = []
                for idx in range(len(f)):
                    wav_length = f[str(idx)].attrs["clean_length"]
                    if wav_length < self.input_size:
                        length = self.output_size
                    else:
                        length =  wav_length - self.start_context_size - self.end_context_size
                    lengths.append(length)
                # Subtract input_size from lengths and divide by hop size to determine number of starting positions
                # Ignore the last to avoid padding
                lengths = [(l // self.output_size) for l in lengths]
            self.start_pos = SortedList(np.cumsum(lengths))
            # Length of the dataset if number of the total frames (which
            # equals last element of the cumulated lengths)
            self.length = self.start_pos[-1]
        return self.length


class EvaluationDataset(Dataset):
    def __init__(self, hdf_file, frame_dur, sr, channels, start_context_dur, end_context_dur):
        """
        This dataset will keep all frames of an utterance in the same batch and 
        clean utterance is not split. 
        :param hdf_file:
        :param sr:
        :param channels:
        """
        super().__init__()

        self.hdf_file = hdf_file
        self.hdf_dataset = h5py.File(hdf_file, 'r')
        self.length = len(self.hdf_dataset)

        self.sr = sr
        self.channels = channels
        self.shapes = None

        self.output_size = int(frame_dur * sr)
        self.start_context_size = int(start_context_dur * sr)
        self.end_context_size = int(end_context_dur * sr)
        self.input_size = self.output_size + self.start_context_size + self.end_context_size
        self.shapes = self._set_shapes()

    def _set_shapes(self):
        shapes = {"input_size" : self.input_size,
                  "output_size" : self.output_size,
                  "start_context_size" : self.start_context_size,
                  "end_context_size" : self.end_context_size}
        return shapes

    def get_shapes(self):
        return self.shapes

    def __getitem__(self, index):
        audio = self.hdf_dataset[str(index)]["noisy"][()].astype(np.float32).squeeze()
        clean = self.hdf_dataset[str(index)]["clean"][()].astype(np.float32).squeeze()

        file_id = self.hdf_dataset[str(index)].attrs["ID"]
        # Check length of audio signal
        audio_length = self.hdf_dataset[str(index)].attrs["length"]
        clean_length = self.hdf_dataset[str(index)].attrs["clean_length"]
        assert(audio_length == clean_length)

        remainder_len = audio_length % self.shapes["output_size"]
        pad_back = 0 if remainder_len == 0 else (self.shapes["output_size"] - remainder_len)
        if pad_back > 0:
            # if pad_back > audio_length:
            #     pad_times = pad_back // audio_length
            #     audio = np.append(audio, [np.tile(audio, pad_times)])
            #     # clean = np.append(clean, [np.tile(clean, pad_times)])
            #     pad_back = pad_back % audio_length

            # audio = np.append(audio, [audio[-pad_back:]])
            audio = np.pad(audio, [(0, pad_back)], mode="constant", constant_values=0.0)
            # clean = np.append(clean, [clean[-pad_back:]])
        
        target_outputs = audio.shape[0]

        # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
        pad_front_context = self.shapes["start_context_size"]
        pad_back_context = self.shapes["end_context_size"]
        audio = np.pad(audio, [(pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)
        # clean = np.pad(clean, [(pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)

        # Iterate over mixture magnitudes, fetch network prediction
        hop_size = self.shapes["output_size"]
        examples = [audio[target_start_pos:target_start_pos + self.shapes["input_size"]]
                    for target_start_pos in range(0, target_outputs, hop_size)]
        # cleans = [clean[target_start_pos:target_start_pos + self.shapes["input_size"]]
        #             for target_start_pos in range(0, target_outputs, hop_size)]

        # return [file_id, np.array(examples), audio_length], [file_id, np.array(cleans), clean_length]
        return [file_id, np.array(examples), audio_length], [file_id, clean, clean_length]

    def __len__(self):
        return self.length
