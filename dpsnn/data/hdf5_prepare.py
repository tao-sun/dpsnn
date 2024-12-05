from tqdm import tqdm
import h5py
import csv
import librosa
import torchaudio
import numpy as np
import torch
import scipy


def load(path, mono=True, mode="numpy", offset=0.0, duration=None):
    y, curr_sr = torchaudio.load(path) # librosa.load(path, sr=None, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)
    # print(f"y orig shape: {y.shape}")

    if len(y.shape) == 1:
        # Expand channel dimension
        y = y[np.newaxis, :]

    if mode == "numpy":
        y = y.numpy()
        # print(f"y shape: {y.shape}")

    return y, curr_sr


def get_samples(csv_path, snr_level="all"):
    reader = csv.reader(open(csv_path, "r"))

    first_row = True
    samples = []
    for row in reader:
        if not row: continue
        if first_row:
            field_lst = row
            first_row = False
            continue

        # Make sure that the current row contains all the fields
        if len(row) != len(field_lst):
            err_msg = (
                    'The row "%s" of the cvs file %s does not '
                    "contain the right number fields (they must be %i "
                    "%s"
                    ")" % (row, csv_path, len(field_lst), field_lst)
            )
            raise ValueError(err_msg)

        row_id = row[0]
        if snr_level == "all" or snr_level == row_id.split("_")[1]:
            attrs = {}
        # Filling the data dictionary
            for i, field in enumerate(field_lst):
                attrs[field] = row[i]
            samples.append(attrs)

    return samples


def create_hdf5(csv_file, hdf_file, sample_rate, channels=1):
    samples = get_samples(csv_file)

    # Create HDF file
    with h5py.File(hdf_file, "w") as f:
        f.attrs["sr"] = sample_rate
        f.attrs["channels"] = channels

        print("Adding audio files to hdf5 (preprocessing)...")
        for idx, example in enumerate(tqdm(samples)):
            # Load mix
            noisy_audio, native_sr = load(example["noisy_wav"], mono=(channels == 1))
            clean_audio, native_sr = load(example["clean_wav"], mono=(channels == 1))
            assert(noisy_audio.shape[1] == clean_audio.shape[1])

            # Add to HDF5 file
            grp = f.create_group(str(idx))
            grp.create_dataset("noisy", shape=noisy_audio.shape, dtype=noisy_audio.dtype, data=noisy_audio)
            grp.create_dataset("clean", shape=clean_audio.shape, dtype=clean_audio.dtype, data=clean_audio)
            grp.attrs["ID"] = example["ID"]
            grp.attrs["length"] = noisy_audio.shape[1]
            grp.attrs["clean_length"] = clean_audio.shape[1]