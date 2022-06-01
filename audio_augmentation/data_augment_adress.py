from setup import torch_setup
import sys
sys.path.append('./ast')
import os
import gc
import torch
import numpy as np
import re
import random
import librosa
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from tqdm import tqdm


import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
from src.models import ASTModel

def id_split(frac=10, print_info=False, cv=True):
    ### Split data into train & val (test is stored separately)
    # Let's aim for a 90/10 split

    # Load the ids of AD subjects into a list
    with open('./audio_filenames_dementia.txt', "r") as clf:
        lines = clf.readlines()
    ids_ad = [re.sub('\n', '', line) for line in lines]

    # Load the ids of Control subjects into a list
    with open('./audio_filenames_control.txt', "r") as clf:
        lines = clf.readlines()
    ids_hc = [re.sub('\n', '', line) for line in lines]

    if print_info:
        print("Number of AD samples:", len(ids_ad), "\nNumber of HC samples:", len(ids_hc))

    # if we don't do cross-validation, we split our ids into train & dev here
    if not cv: 
        for i, id_list in enumerate([ids_ad, ids_hc]):
            split = len(id_list) // frac # number of ids to include the validation set
            # random.shuffle(id_list)

            # perform split
            val_ids = id_list[:split]
            train_ids = id_list[split:]

            # save the output as a dictionary
            if i == 0:
                ad_id_dict = dict()
                ad_id_dict['train'], ad_id_dict['val']= train_ids, val_ids
            else:
                hc_id_dict = dict()
                hc_id_dict['train'], hc_id_dict['val']= train_ids, val_ids

        if print_info:
            print("Number of AD samples for validation", len(ad_id_dict['val']))
            print("Number of HC samples for validation", len(hc_id_dict['val']))

        return ids_ad, ids_hc, ad_id_dict, hc_id_dict
    
    return ids_ad, ids_hc


from scipy import signal
from scipy.signal import get_window

aug8 = nas.FrequencyMaskingAug()
aug9 = nas.TimeMaskingAug()
    
    
def vc_to_mel(x, augmentation_type="notspec"):

    S = librosa.power_to_db(
                    librosa.feature.melspectrogram(
                        x.numpy().astype(np.float32)/80, sr=sr, n_fft=n_fft, 
                        hop_length=hop_length),ref=np.max)
    if augmentation_type=="specaugment":
        S = aug8.augment(S)
        S = aug9.augment(S)
    return S

def segment_mp3_tensor(augment, logmel, autovc_data, audio_path, audio_trg_path, duration, spacing = None, augmentation_type = 'random'):
    """
    This function loads an mp3 or wav file, converts it to a tensor, splits up the 
    tensor into segments of a desired length and returns these in a list.

    Arguments
        audio: a string containing the path, we assume mp3
        duration: required length of a segment in seconds
        spacing: step size between segments, in seconds
    
    Returns
        segments: a list of tensors
    """
    segments = []
    audio, notimp = sf.read(audio_path)
    if autovc_data:
        audio = librosa.resample(audio, 16000, 44100)
    audio = torch.Tensor(audio)

    audio_len = len(audio)
    start_pt = 0
    end_pt = duration*sr
    while end_pt < audio_len:
        segments.append(audio[start_pt:end_pt])
        if spacing is None: # non-overlapping segments
            start_pt += duration*sr
            end_pt += duration*sr
        else:
            start_pt += spacing*sr
            end_pt += spacing*sr

    random_aug_size = 1
    if augment:
        # for a specific kind of augmentation, just create a vector of 
        # the number representing the desired method
        if augmentation_type == 'random':
            aug_num = random.choices(population=list(range(8)), k=len(segments))
            if random_aug_size == 2:
                aug_num2 = random.choices(population=list(range(9)), k=len(segments))
            elif random_aug_size == 4:
                aug_num2 = random.choices(population=list(range(9)), k=len(segments))
                aug_num3 = random.choices(population=list(range(9)), k=len(segments))
                aug_num4 = random.choices(population=list(range(9)), k=len(segments))
        elif augmentation_type == 'masking':
            aug_num = [0 for i in range(len(segments))]
        elif augmentation_type == 'loudness':
            aug_num = [1 for i in range(len(segments))]
        elif augmentation_type == 'noise':
            aug_num = [2 for i in range(len(segments))]
        elif augmentation_type == 'pitch':
            aug_num = [3 for i in range(len(segments))]
        elif augmentation_type == 'vtlp':
            aug_num = [4 for i in range(len(segments))]
        elif augmentation_type == 'shift':
            aug_num = [5 for i in range(len(segments))]
        elif augmentation_type == 'speed':
            aug_num = [6 for i in range(len(segments))]
        elif augmentation_type == 'normalisation':
            aug_num = [7 for i in range(len(segments))]
        if not augmentation_type == 'specaugment':
            segments_aug = [random_augmentation(segments[i], aug_num=aug_num[i]) for i in list(range(len(segments)))]
        else:
            segments_aug = [segments[i] for i in list(range(len(segments)))]
        if augmentation_type == 'random':    
            if random_aug_size == 2:
                segments_aug2 = [random_augmentation(segments[i], aug_num=aug_num2[i]) for i in list(range(len(segments)))]
                segments_aug = segments_aug + segments_aug2
            elif random_aug_size == 4:
                segments_aug2 = [random_augmentation(segments[i], aug_num=aug_num2[i]) for i in list(range(len(segments)))]
                segments_aug3 = [random_augmentation(segments[i], aug_num=aug_num3[i]) for i in list(range(len(segments)))]
                segments_aug4 = [random_augmentation(segments[i], aug_num=aug_num4[i]) for i in list(range(len(segments)))]
                segments_aug = segments_aug + segments_aug2 + segments_aug3 + segments_aug4


        segments_aug = [torch.tensor(segments_aug[i]) for i in list(range(len(segments_aug)))]

        if logmel:
            # waveform-level augmentation
            segments_aug = [torch.tensor(vc_to_mel(segment, augmentation_type)) for segment in segments_aug]

    if logmel:
        segments = [torch.tensor(vc_to_mel(segment)) for segment in segments]
    
    if augment:
        return segments, segments_aug
    else:
        return segments

sr_a = 44100
aug0 = naa.MaskAug(sampling_rate=sr_a, zone=(0,1), coverage=0.3)
aug1 = naa.LoudnessAug(zone=(0,1), factor=(0.3, 3))
aug3 = naa.PitchAug(sampling_rate=sr_a, zone=(0,1))
aug4 = naa.VtlpAug(zone=(0,1), sampling_rate=sr_a, factor=(0.5,3), coverage=1) # needs padding after!
aug5 = naa.ShiftAug(sampling_rate=sr_a, duration=0.5)
aug6 = naa.SpeedAug(factor=(0.5,1)) # needs cropping after!
aug7 = naa.NormalizeAug(coverage=1, method='max')
aug8 = nas.FrequencyMaskingAug()
aug9 = nas.TimeMaskingAug()

def random_augmentation(segment, aug_num):
    """
    There are 8 augmentations
    0 - Masking
    1 - Loudness
    2 - Noise
    3 - Pitch
    4 - VTLP
    5 - Shift
    6 - Speed
    7 - Normalization
    """
    dur_a = 705600
    if aug_num == 0: # masking
        aug_segment = aug0.augment(segment.detach().numpy())
    
    elif aug_num == 1: # loudness
        aug_segment = aug1.augment(segment.detach().numpy())
    
    elif aug_num == 2: # noise
        noise = np.random.randn(len(segment))
        aug_segment = segment.detach().numpy() + 0.002 * noise

    elif aug_num == 3: # pitch
        aug_segment = aug3.augment(segment.detach().numpy())

    elif aug_num == 4: # vtlp
        aug_segment = aug4.augment(segment.detach().numpy())
        len_to_add = dur_a -  len(aug_segment)
        zeros = np.zeros(len_to_add)
        aug_segment = np.concatenate((aug_segment, zeros))
    
    elif aug_num == 5: # shift
        aug_segment = aug5.augment(segment.detach().numpy())
    
    elif aug_num == 6: # speed
        aug_segment = aug6.augment(segment.detach().numpy())
        aug_segment = aug_segment[:dur_a]
    
    elif aug_num == 7: # normalization
        aug_segment = aug7.augment(segment.detach().numpy())

    elif aug_num == 8: # specaugment
        aug_segment = librosa.feature.melspectrogram(segment.detach().numpy())
        aug_segment = aug8.augment(aug_segment)
        aug_segment = aug9.augment(aug_segment)
        aug_segment = librosa.feature.inverse.mel_to_audio(aug_segment)

    return aug_segment


def segment_samples(use_autovc, file_lst, file_lst2, foldername, duration, spacing, sr, n_fft, 
                    hop_length, AD_flag, train_data = [], train_labels=[], 
                    val_data=[], val_labels=[], logmel=True, ids_dict=None, augment=False, augmentation_type='random'):
    """
    ids_dict is only specified when we don't perform cross-validation, so the 
    train & validation split is already given. So in a way, the presence or
    absence of ids_dict acts as a cv = True / False flag.
    """


    for filename in tqdm(file_lst):
        path = os.path.join(foldername, f'{filename}.wav')
        #alican vc
        filename_trg = random.choice(file_lst)
        path_trg = os.path.join(foldername, f'{filename_trg}.wav')

        if augment:
            segments, segments_aug = segment_mp3_tensor(augment, logmel, False, path, path_trg, duration, spacing, augmentation_type)
        else:
            segments = segment_mp3_tensor(augment, logmel, False, path, path_trg, duration, spacing, augmentation_type)


        if ids_dict:
            if filename in ids_dict['train']:
                train_data.extend(segments)
                train_labels.extend([AD_flag for segment in segments])
                if augment:
                    train_data.extend(segments_aug)
                    train_labels.extend([AD_flag for segment in segments_aug])
            elif filename in ids_dict['val']:
                val_data.extend(segments)
                val_labels.extend([AD_flag for segment in segments])
                if augment:
                    val_data.extend(segments_aug)
                    val_labels.extend([AD_flag for segment in segments_aug])

####FOR AUTOVC AUGMENTED DATASET
    if use_autovc:
        for filename in file_lst2:
            path = os.path.join(foldername + '_vc', f'{filename}')
            filename_trg = random.choice(file_lst)
            path_trg = os.path.join(foldername + '_vc', f'{filename_trg}')

            if augment:
                segments, segments_aug = segment_mp3_tensor(augment, logmel, True, path, path_trg, duration, spacing, augmentation_type)
            else:
                segments = segment_mp3_tensor(augment, logmel, True, path, path_trg, duration, spacing, augmentation_type)

            train_data.extend(segments)
            train_labels.extend([AD_flag for segment in segments])
            if augment:
                train_data.extend(segments_aug)
                train_labels.extend([AD_flag for segment in segments_aug])

    if ids_dict:
        return train_data, train_labels, val_data, val_labels



def segment_test_data(use_autovc, test_path, logmel=True, duration=16, augment=False):

    segment_dict_id = dict()
    dataset, labels = [], []

    with open(f'{PATH}/test_labels.txt', "r") as clf:
        lines = clf.readlines()
    ids_and_labels = [line.strip() for line in lines]
    ids = [item[:4] for item in ids_and_labels]

    test_ids_ad, test_ids_hc = [], []

    for id in ids_and_labels:
        if int(id[-1]) == 1:
            test_ids_ad.append(id[:4])
        elif int(id[-1]) == 0:
            test_ids_hc.append(id[:4])

    test_seg_id_a = []
    for file in ids:
        path = os.path.join(test_path, f'{file}.wav')
        filename_trg = random.choice(ids)
        path_trg = os.path.join(test_path, f'{filename_trg}.wav')
        segments = segment_mp3_tensor(augment, logmel, False, path, path_trg, duration, spacing)
        test_seg_id_a = test_seg_id_a + [file for i in range(len(segments))]

        segment_dict_id[file] = segments
        dataset.extend(segments)

        if file in test_ids_ad:
            labels.extend([1 for segment in segments])
        elif file in test_ids_hc:
            labels.extend([0 for segment in segments])
    assert (len(dataset)==len(labels))

    return dataset, labels, segment_dict_id, test_seg_id_a



class Dataset_AST(Dataset):

    def __init__(self, ids, labels):
        ids = [torch.tensor(id) for id in ids]
        self.x_train = ids
        self.y_train = torch.tensor(labels)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, item):
        return self.x_train[item], self.y_train[item]




DEVICE = torch_setup()
PATH = "./"
EPOCHS = 2 
sr = 44100 
n_fft = 1102
hop_length = 441 
lr = 1e-06
batch_size = 8 
duration = 10 
spacing = 2
ad_path = "./ad_audio_adress"
hc_path = "./hc_audio_adress"
test_path = "./test_audio_adress"
use_autovc = True
use_augment = False
augmentation_type= 'autovc' #["random", "masking", "shift", "loudness", "noise", "pitch", "vtlp", "speed", "normalisation", "specaugment"]
random_aug_size = 1

ids_ad, ids_hc, ad_id_dict, hc_id_dict = id_split(10, print_info=True, cv=False)
file_lst2_ad = os.listdir("./ad_audio_adress_vc/")
file_lst2_hc = os.listdir("./hc_audio_adress_vc/")

train_data, train_labels, val_data, val_labels = segment_samples(use_autovc, ids_ad, file_lst2_ad, ad_path, 16, 2, sr, n_fft, hop_length, AD_flag=1, logmel=True, ids_dict=ad_id_dict, augment=use_augment, augmentation_type=augmentation_type)
train_data, train_labels, val_data, val_labels = segment_samples(use_autovc, ids_hc, file_lst2_hc, hc_path, 16, 2, sr, n_fft, hop_length, 0, train_data, train_labels, val_data, val_labels, ids_dict = hc_id_dict, augment=use_augment, augmentation_type=augmentation_type)
gc.collect()
torch.cuda.empty_cache()
train_data.extend(val_data)
train_labels.extend(val_labels)

for i in tqdm(range(len(train_data))):
    file = train_data[i].numpy()
    file_path = './data_' + augmentation_type + '/' + str(i) + '.csv'
    np.savetxt(file_path, file, delimiter=',')
np.savetxt('./train_labels_' + augmentation_type, train_labels, delimiter=',')