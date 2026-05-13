# Data Processing Functions

import os
import shutil
import librosa
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Data input and output directories:
train_data_path = "../Data/train"
test_data_path = "../Data/test"
sample_input_path = "../Data/sandbox"
proc_train_data_path = "../Data_Proc/train"
proc_test_data_path = "../Data_Proc/test"
eval_train_data_path = "../Eval_Data/train"
eval_test_data_path = "../Eval_Data/test"

def get_all_au_test_data(features, data_path=test_data_path):
    all_features_list = []
    all_files_list = []
    for directory_path, _, files in os.walk(data_path):
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            if ((os.path.isfile(file_path)) and (os.path.splitext(file_path)[1] == ".au")):
                file_features = []
                for feature in features:
                    feature_load_func = feature_load_selector(feature)
                    feature_data = feature_load_func(file_path, hop_length=512, n_mfcc=13)
                    rows, cols = feature_data.shape
                    feature_data = feature_data[:,:1200]
                    feature_data_flat = feature_data.reshape(-1)
                    feature_data_flat = feature_data_flat.reshape(1,-1)
                    file_features.append(feature_data_flat)

                file_features_combined = np.hstack(file_features)
                all_features_list.append(file_features_combined)
                all_files_list.append(filename)
                
    return all_features_list, all_files_list


def split_spec_data_for_CNN_test_eval(data_path=proc_train_data_path):
    class_list = []
    path_list = []
    for directory_path, dirs, files in os.walk(data_path):
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            class_name = os.path.basename(os.path.normpath(directory_path))
            if ((os.path.isfile(file_path)) and (os.path.splitext(file_path)[1] == ".png")):
                class_list.append(class_name)
                path_list.append(file_path)
    return class_list, path_list

def create_class_dirs_in_eval_dirs(class_list, train_output_dir=eval_train_data_path, 
                                   test_output_dir=eval_test_data_path):
    unique_class_names = list(set(class_list))
    for class_name in unique_class_names:
        train_class_dir = Path(os.path.join(train_output_dir, class_name))
        test_class_dir = Path(os.path.join(test_output_dir, class_name))
        train_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)

def train_test_split_CNN_data(class_list, path_list):
    # Split training data
    train_test_sets = train_test_split(path_list, class_list, train_size=.8, random_state=99, stratify=class_list)
    trainpath_list, testpath_list, trainclass_list, testclass_list = train_test_sets
    return trainpath_list, testpath_list, trainclass_list, testclass_list 

def move_eval_files(path_list, class_list, output_dir):
    zip_list = zip(path_list, class_list)
    for sourcefile,classname in zip_list:
        filename = os.path.basename(sourcefile)
        destination = os.path.join(output_dir, classname, filename)
        shutil.copyfile(sourcefile, destination)
    return

def proc_all_au_test_data(features, data_path=test_data_path, output_dir=proc_test_data_path, hop_length=512, n_mfcc=13, n_fft=2048):
    for directory_path, _, files in os.walk(data_path):
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            filebase = os.path.splitext(filename)[0]
            fileoutput = os.path.join(output_dir, filebase + ".png")
            if ((os.path.isfile(file_path)) and (os.path.splitext(file_path)[1] == ".au")):
                group_features = []
                for feature in features:
                    feature_load_func = feature_load_selector(feature)
                    feature_data = feature_load_func(file_path, hop_length=hop_length, n_mfcc=n_mfcc, n_fft=n_fft)
                    feature_data = feature_data[:,:1200]
                    group_features.append(feature_data)

                features_combined = np.vstack(group_features)

                norm_feature = (features_combined - features_combined.min())/(features_combined.max()-features_combined.min()) * 255.0
                feature_uint8 = norm_feature.astype(np.uint8)
                image_array = Image.fromarray(feature_uint8, 'L')
                image_array.save(fileoutput)
                
    return

def proc_all_au_data(features, data_path=train_data_path, output_dir=proc_train_data_path, hop_length=512, n_mfcc=13, n_fft=2048):
    for directory_path, _, files in os.walk(data_path):
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            print(f"File path = {file_path}")
            class_name = os.path.basename(os.path.normpath(directory_path))
            output_path = os.path.join(output_dir, class_name)
            filebase = os.path.splitext(filename)[0]
            fileoutput = os.path.join(output_path, filebase + ".png")
            if ((os.path.isfile(file_path)) and (os.path.splitext(file_path)[1] == ".au")):
                group_features = []
                for feature in features: 
                    feature_load_func = feature_load_selector(feature)
                    feature_data = feature_load_func(file_path, hop_length=hop_length, n_mfcc=n_mfcc, n_fft=n_fft)
                    feature_data = feature_data[:,:1200]
                    group_features.append(feature_data)

                features_combined = np.vstack(group_features)

                norm_feature = (features_combined - features_combined.min())/(features_combined.max()-features_combined.min()) * 255.0
                feature_uint8 = norm_feature.astype(np.uint8)
                os.makedirs(output_path, exist_ok=True)
                image_array = Image.fromarray(feature_uint8, 'L')
                image_array.save(fileoutput)
    return

def proc_all_au_data_rgb(features, data_path=train_data_path, output_dir=proc_train_data_path, hop_length=512, n_mfcc=13, n_fft=2048):
    for directory_path, _, files in os.walk(data_path):
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            class_name = os.path.basename(os.path.normpath(directory_path))
            output_path = os.path.join(output_dir, class_name)
            filebase = os.path.splitext(filename)[0]
            fileoutput = os.path.join(output_path, filebase + ".png")
            if ((os.path.isfile(file_path)) and (os.path.splitext(file_path)[1] == ".au")):
                group_features = []
                for feature in features: 
                    feature_load_func = feature_load_selector(feature)
                    feature_data = feature_load_func(file_path, hop_length=hop_length, n_mfcc=n_mfcc, n_fft=n_fft)
                    feature_data = feature_data[:,:1200]
                    group_features.append(feature_data)

                features_combined = np.vstack(group_features)

                norm_feature = (features_combined - features_combined.min())/(features_combined.max()-features_combined.min())
                cmap = plt.get_cmap('magma')
                colored_feature_rgba = cmap(norm_feature)
                colored_feature_rgb = colored_feature_rgba[:,:,:3]
                colored_feature_rgb_uint8 = (colored_feature_rgb * 255).astype(np.uint8)
                os.makedirs(output_path, exist_ok=True)
                image_array = Image.fromarray(colored_feature_rgb_uint8, 'RGB')
                image_array.save(fileoutput)
    return

def proc_all_au_test_data_rgb(features, data_path=test_data_path, output_dir=proc_test_data_path, hop_length=512, n_mfcc=13, n_fft=2048):
    for directory_path, _, files in os.walk(data_path):
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            filebase = os.path.splitext(filename)[0]
            fileoutput = os.path.join(output_dir, filebase + ".png")
            if ((os.path.isfile(file_path)) and (os.path.splitext(file_path)[1] == ".au")):
                group_features = []
                for feature in features:
                    feature_load_func = feature_load_selector(feature)
                    feature_data = feature_load_func(file_path, hop_length=hop_length, n_mfcc=n_mfcc, n_fft=n_fft)
                    feature_data = feature_data[:,:1200]
                    group_features.append(feature_data)

                features_combined = np.vstack(group_features)

                norm_feature = (features_combined - features_combined.min())/(features_combined.max()-features_combined.min())
                cmap = plt.get_cmap('magma')
                colored_feature_rgba = cmap(norm_feature)
                colored_feature_rgb = colored_feature_rgba[:,:,:3]
                colored_feature_rgb_uint8 = (colored_feature_rgb * 255).astype(np.uint8)
                image_array = Image.fromarray(colored_feature_rgb_uint8, 'RGB')
                image_array.save(fileoutput)
                
    return

def get_all_au_data(features, data_path=train_data_path, hop_length=512, n_mfcc=13):
    all_features_list = []
    for directory_path, _, files in os.walk(data_path):
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            if ((os.path.isfile(file_path)) and (os.path.splitext(file_path)[1] == ".au")):
                file_features = []
                for feature in features: 
                    feature_load_func = feature_load_selector(feature)
                    feature_data = feature_load_func(file_path, hop_length=hop_length, n_mfcc=n_mfcc)
                    rows, cols = feature_data.shape
                    feature_data = feature_data[:,:1200]
                    feature_data_flat = feature_data
                    feature_data_flat = feature_data_flat.reshape(1,-1)
                    file_features.append(feature_data_flat)

                file_features_combined = np.hstack(file_features)
                all_features_list.append(file_features_combined)
         
    return all_features_list

def get_all_au_data_labels(data_path=train_data_path):
    Y = []
    for directory_path, _, files in os.walk(data_path):
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            genre = os.path.basename(directory_path)
            if ((os.path.isfile(file_path)) and (os.path.splitext(file_path)[1] == ".au")):
                Y.append(genre)
                
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    Y = np.array(Y)
    Y = Y.reshape(-1,1)
    return Y, encoder

def load_au_as_mfcc(file_path, **kwargs):
    hop_length = kwargs['hop_length']
    n_mfcc = kwargs["n_mfcc"]
    # y is audio time series
    # sr is sampling rate
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
    return mfccs

def load_au_as_melspec(file_path, **kwargs):
    hop_length = kwargs['hop_length']
    n_fft = kwargs["n_fft"]
    y, sr = librosa.load(file_path)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    dB_melspec = librosa.power_to_db(mel_spec)
    return dB_melspec

def load_au_as_chromas(file_path, **_):
    y, sr = librosa.load(file_path)
    chromas = librosa.feature.chroma_stft(y=y, sr=sr)
    return chromas

def load_au_as_spectral_contrast(file_path, **_):
    y, sr = librosa.load(file_path)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return spec_contrast

def load_au_as_zero_crossings(file_path, **_):
    y, sr = librosa.load(file_path)
    zero_crossings = librosa.feature.zero_crossing_rate(y=y)
    return zero_crossings

def load_au_as_stft(file_path,  **kwargs):
    hop_length = kwargs['hop_length']
    n_fft = kwargs["n_fft"]
    y, sr = librosa.load(file_path)
    stft_coeffs = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    power_spectral_density = np.abs(stft_coeffs) ** 2
    dB_power_spec = librosa.power_to_db(power_spectral_density)
    return dB_power_spec

def plot_feature(feature_type, feature):
    plt.figure(figsize=(10,4))
    librosa.display.specshow(feature, x_axis='time')
    plt.colorbar()
    plt.title(feature_type)
    plt.tight_layout()
    plt.show()

def feature_load_selector(selected_feature):
    match selected_feature:
        case "mfcc":
            feature_load_func = load_au_as_mfcc
        case "chroma":
            feature_load_func = load_au_as_chromas
        case "spectral contrast":
            feature_load_func = load_au_as_spectral_contrast
        case "zero crossings":
            feature_load_func = load_au_as_zero_crossings
        case "stft":
            feature_load_func = load_au_as_stft
        case "compound":
            feature_load_func = compound_feature_extraction
        case "melspec":
            feature_load_func = load_au_as_melspec
        case _:
            print("nope")
    return feature_load_func

def debug_decode_au_labels(labels, encoder):
    # This is specifically for decoding labels as shaped
    # by get_all_au_labels() function
    labels = labels.reshape(-1)
    classes = encoder.inverse_transform(labels)
    return classes

def normalize_feature(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    normalized_data = (data-min_vals) / (max_vals-min_vals)
    return normalized_data


def compound_feature_extraction(file_path, **kwargs):
    hop_length = kwargs['hop_length']
    n_mfcc = kwargs["n_mfcc"]

    # Load audio data
    y, sr = librosa.load(file_path, sr=None)
    
    # Decompose into harmonic and percussive audio components
    y_harmonic, y_percussive = librosa.effects.hpss(y=y)

    # Extract the beat track (based on frame size from selected hop length)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
    
    # Normalize MFCCs data
    norm_mfcc = normalize_feature(mfcc)
    
    # Extract MFCC Delta
    mfcc_delta = librosa.feature.delta(mfcc)
    # Normalize MFCC Delta data
    norm_mfcc_delta = normalize_feature(mfcc_delta)

    # Stack MFCC and MFCC Delta data
    stacked_mfcc_and_delta = np.vstack([mfcc, mfcc_delta])
    # Normalize stacked MFCC and MFCC Delta data
    norm_stacked_mfcc_and_delta = np.vstack([norm_mfcc, norm_mfcc_delta])
    # Use only normalized stacked data subsequently
    stacked_mfcc_and_delta = norm_stacked_mfcc_and_delta
    
    # Sync stacked MFCC data to beat frames (using default aggregation of "average")
    beat_mfcc_delta = librosa.util.sync(stacked_mfcc_and_delta, beat_frames) # type: ignore
    
    # Extract chromatographic data
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    
    # Sync chromagram to beat frames, use median to aggregate
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median) # type: ignore
    
    # Stack synced chromagram to synced stacked MFCC and MFCC delta
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
    return beat_features