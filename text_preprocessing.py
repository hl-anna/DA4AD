import os
import pylangacq
import re
from text_augmentation import random_augment

def data_to_str(file_lst:list, path:str, AD_flag, model_type='bert', 
                augment=True, rand_augment='min_1', aug_bt=None, sd_min=0, 
                sd_max=2, aug_vect=None):
    """
    This function performs pre-processing of the transcripts stored in CHAT 
    files as well as data augmentation (if required).

    Parameters:
        - file_lst: a list of file ids (i.e. filenames without the '.cha')
        - path: the path to where the transcript files are
        - AD_flag: label to assign to samples (data to be processed by class)
        - model_type: either 'bert' or 't5' 
        - augment: Boolean flag
        - rand_augment: if random aug. is performed, can be 'min_1' or 'any'
        - aug_bt: if backtranslation is to be performed as part of augmentation,
                  a backtranslation class from NLPAug must be given
        - sd_min: if random deletion is performed as part of augmentation, the 
                  the minimum number of sentences to be deleted
        - sd_max: if random deletion is performed as part of augmentation, the 
                  the max number of sentences to be deleted

    Returns:
        - dataset : a list of strings, the pre-processed transcripts
        - labels : a list of strings or ints, the labels
        - aug_dataset : a list of strs/tuples/Nones, the augmented transcripts

    """

    dataset, labels, aug_dataset = [], [], []

    for file in file_lst:
        if model_type == 'bert':
            transcript = []
        elif model_type == 't5':
            transcript = ["binary classification:"]
            
        filepath = os.path.join(path, file+".cha")
        data = pylangacq.read_chat(filepath)
        data = data.words(participants="PAR", by_utterances=True)

        for sent in data:
            sent_new = re.split(" |_|-", " ".join(sent).lower())
            if model_type == 'bert':
                sent_new.append("[SEP]")
            elif model_type == 't5':
                sent_new.append("</s>")
            transcript.append(" ".join(sent_new))
        
        if augment:
            aug_tr = random_augment(transcript, aug_bt=aug_bt, sd_min=sd_min, 
                                    sd_max=sd_max, model_type=model_type, 
                                    augment_type=rand_augment, aug_vect=aug_vect)
            
            assert type(aug_tr) == str or type(aug_tr) == tuple or type(aug_tr) == None
            aug_dataset.append(aug_tr)

        transcript = " ".join(transcript)
        assert type(transcript) == str
        
        if model_type == 'bert':
            transcript = transcript[:-5]
        elif model_type == 't5':
            transcript = transcript[:-4]
        
        dataset.append(transcript)
        labels.append(AD_flag)

    return dataset, labels, aug_dataset


def test_data_to_str(id_path, data_path, model_type='bert', return_dict=False):
    """
    This function performs pre-processing of the test set transcripts.

    Parameters:
    id_path: a (complete) path to a 'txt' file with the file names, i.e. ids
             NOTE: these must be saved one per line, and together with the
                   label. 
                   e.g. if the id is S123 and label is 0, the corresponding
                        line in the text file will be 'S123_0'
    data_path: path to the folder where the test transcripts are saved
    model_type: either 'bert' or 't5'

    Returns:
        - dataset: a list of strings, the transcripts
        - labels: a list of strings or ints, the labels
    """

    with open(id_path, "r") as clf:
        lines = clf.readlines()
    ids_and_labels = [line.strip() for line in lines]
    ids = [item[:4] for item in ids_and_labels]

    test_ids_ad, test_ids_hc = [], []

    for id in ids_and_labels:
        if int(id[-1]) == 1:
            test_ids_ad.append(id[:4])
        elif int(id[-1]) == 0:
            test_ids_hc.append(id[:4])

    dataset, labels = [], []
    dict_by_id = dict()

    for file in ids:
        
        if model_type == 'bert':
            transcript = []
        elif model_type == 't5':
            transcript = ["binary classification:"]
            
        filepath = os.path.join(data_path, file+".cha")
        data = pylangacq.read_chat(filepath)
        data = data.words(participants="PAR", by_utterances=True)

        for sent in data:
            sent_new = re.split(" |_|-", " ".join(sent).lower())
            if model_type == 'bert':
                sent_new.append("[SEP]")
            elif model_type == 't5':
                sent_new.append("</s>")
            transcript.append(" ".join(sent_new))
            
        transcript = " ".join(transcript)
        assert type(transcript) == str

        if model_type == 'bert':
            transcript = transcript[:-5]
        elif model_type == 't5':
            transcript = transcript[:-4]
            
        dataset.append(transcript)
        dict_by_id[file] = transcript
        
        if file in test_ids_ad:
            if model_type == 'bert':
                labels.append(1)
            elif model_type == 't5':
                labels.append("1")
        elif file in test_ids_hc:
            if model_type == 'bert':
                labels.append(0)
            elif model_type == 't5':
                labels.append("0")

    assert (len(dataset)==len(labels))

    if return_dict:
        return dataset, labels, dict_by_id
    else:
        return dataset, labels


def data_to_str_custom(file_lst:list, path:str, AD_flag, model_type='bert', 
                       augment=True, rand_augment='min_1', aug_bt=None, 
                       sd_min=0, sd_max=2):
    """
    This function performs pre-processing of the transcripts stored in CHAT 
    files on a more granual level, as well as data augmentation (if required).
    Note that the 'data_to_str' function above is used by default.

    Parameters:
        - file_lst: a list of file ids (i.e. filenames without the '.CHAT')
        - path: the path to where the transcript files are
        - AD_flag: label to assign to samples (data to be processed by class)
        - model_type: either 'bert' or 't5' 
        - augment: Boolean flag
        - rand_augment: if random aug. is performed, can be 'min_1' or 'any'
        - aug_bt: if backtranslation is to be performed as part of augmentation,
                  a backtranslation class from NLPAug must be given
        - sd_min: if random deletion is performed as part of augmentation, the 
                  the minimum number of sentences to be deleted
        - sd_max: if random deletion is performed as part of augmentation, the 
                  the max number of sentences to be deleted

    Returns:
        - dataset : a list of strings, the pre-processed transcripts
        - labels : a list of strings or ints, the labels
        - aug_dataset : a list of strs/tuples/Nones, the augmented transcripts
    """

    dataset, labels, aug_dataset = [], [], []

    for file in file_lst:
        if model_type == 'bert':
            transcript = []
        elif model_type == 't5':
            transcript = ["binary classification:"]
        filepath = os.path.join(path, file+".cha")
        data = pylangacq.read_chat(filepath)
        
        for utterance in data.utterances():
            if 'PAR' in utterance.tiers.keys():
                sent = utterance.tiers['PAR']

                pattern = "\[.*?\] " # remove anything in square brackets
                sent_new = re.sub(pattern, "", sent)

                pattern_3 = 'xxx '
                sent_new = re.sub(pattern_3, '', sent_new)
                
                pattern_4 = "[^A-Za-z0-9 .'?!]" # remove unwanted characters
                sent_new = re.sub(pattern_4, "", sent_new).lower()
                
                sent_new = re.split(" |-|'", sent_new)
                
                # remove numeric code from the end of each sentence
                sent_new.pop()
                if model_type == 'bert':
                    sent_new.append("[SEP]")
                elif model_type == 't5':
                    sent_new.append("</s>")
                sent_new = " ".join(sent_new).strip()
                if sent_new[:2] == ". ":
                    sent_new = sent_new[2:]

                transcript.append(sent_new)
        
        if augment:
            aug_tr = random_augment(transcript, aug_bt=aug_bt, sd_min=sd_min, 
                                    sd_max=sd_max, model_type=model_type, 
                                    augment_type=rand_augment)
            
            assert type(aug_tr) == str or type(aug_tr) == tuple or type(aug_tr) == None
            aug_dataset.append(aug_tr)
        
        transcript = " ".join(transcript) # we'll use this for augmentation
        pattern_5 = '( \.)+'
        transcript = re.sub(pattern_5, " .", transcript)
        assert type(transcript) == str
        
        if model_type == 'bert':
            transcript = transcript[:-5]
        elif model_type == 't5':
            transcript = transcript[:-4]
        
        dataset.append(transcript)
        labels.append(AD_flag)

    return dataset, labels, aug_dataset