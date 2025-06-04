import json
import os
import math
import librosa
import numpy as np


DATASET_PATH = "d:\\ΔΙπλωματική\\DATAMAYBEFINAL1"
JSON_PATH = "d:\\ΔΙπλωματική\\data_2.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def extract_mel_spectrogram(y, sr, n_mels=128, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def pad_truncate_spectrogram(spectrogram, max_len):
    if spectrogram.shape[1] < max_len:
        pad_width = max_len - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spectrogram = spectrogram[:, :max_len]
    return spectrogram

def save_features(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, n_mels=128, num_segments=5):
    """Extracts MFCCs and Mel Spectrograms from music dataset and saves them into a json file along with genre labels."""

    # store mapping, labels, MFCCs, and Mel Spectrograms
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [],
        "mel_spectrogram": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    num_mel_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                y = signal

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract MFCC
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # extract Mel Spectrogram
                    mel_spectrogram = extract_mel_spectrogram(y=signal[start:finish], sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
                    mel_spectrogram = pad_truncate_spectrogram(mel_spectrogram, num_mel_vectors_per_segment).T

                    # store only features with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment and len(mel_spectrogram) == num_mel_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["mel_spectrogram"].append(mel_spectrogram.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))

    # save features to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_features(DATASET_PATH, JSON_PATH, num_segments=10)
