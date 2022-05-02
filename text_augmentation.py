import random
import numpy as np
from sklearn.utils import shuffle

def random_deletion(text, min_d=0, max_d=2):
    """
    Deletes between 'min' and 'n' randomly selected elements from a list 
    (here a list of strings)
    
    Returns:
        string
    """

    indices = random.sample(list(range(len(text))), random.randint(min_d,max_d))
    indices.sort(reverse=True)

    for idx in indices:
        text.pop(idx)

    return text

def mixup(lst_mixup, label, seed):
    """
    This function performs the 'Mixup' augmentation operation.
    
    Parameters:
        - lst_mixup: list of tuples, the data to be mixed up
        - label: mixup is performed only between samples of the same class 
        - seed : random seed

    Returns
    mixup_data: list of strings
    mixup_labels : list of ints or strings
    """
    
    h1 = [sample[0] for sample in lst_mixup]
    h2 = [sample[1] for sample in lst_mixup]
    assert len(h1) == len(h2)

    h1 = shuffle(h1, random_state=seed)
    mixup_data = [h1[i]+h2[i] for i in list(range(len(h1)))]
    mixup_labels = [label for item in mixup_data]

    return mixup_data, mixup_labels

def random_augment(transcript:str, aug_bt=None, aug_vect=None, sd_min:int=0, 
                   sd_max:int=2, model_type='bert', augment_type = 'min_1'):
    """
    This function applies anything from none to all of our three augmentation
    methods to a given transcript.
    It does this by either generating or getting a list of three integers 
    (either a 0 or a 1, these will be used as booleans), where each integer 
    represents one of the augmentation methods and whether it is applied (1) 
    or not (0).

    aug_vect[0] ~ deletion, aug_vect[1] ~ backtransl., aug_vect[2] ~ mixup
    
    So for example, [0, 1, 1] means that backtranslation & mixup are applied,
    but not random deletion.
    
    Parameters:
        - transcript: a string to be augmented
        - aug_bt: provide a backtranslation class object (from NLPaug) for 
                  backtranslation augmentation to be applied
        - aug_vect: to apply a specific augmentation only (not random),
                    specify the required list, e.g. [0, 1, 1]
        - model_type: either 'bert' or 't5', depending on the model the data is
                      used to train
        - augment_type: to apply random aug., select either 'min_1' or 'any'
        - sd_min: if sentence deletion is used, specifies the minimum number
                  sentences to be deleted
        - sd_max: if sentence deletion is used, specifies the max number of
                  sentences to be deleted

    Returns:
        None - if no augmentation has been applied
        String - (the augmented transcript) if mixup has not been applied
        Tuple - of 2 strings (transcript halves) if mixup has been applied
    """
    
    assert model_type == 'bert' or model_type == 't5'

    if aug_vect == None:
        assert augment_type == 'any' or augment_type == 'min_1'
        aug_vect = np.random.randint(0, 2, size=3)

    if augment_type == 'any':
        if sum(aug_vect) == 0: # no augmentation, this happens in ~ 1/8 cases
            return None
    elif augment_type == 'min_1':
        while sum(aug_vect) == 0:
            aug_vect = np.random.randint(0, 2, size=3)

    # apply augmentation based on the aug_vect provided or generated
    
    if aug_vect[2] == 1: # apply mixup
        split = len(transcript) // 2

        if aug_vect[0] == 1: # apply deletion
            transcript_half1 = random_deletion(transcript[:split], 
                                               min_d=sd_min, max_d=sd_max)
            transcript_half2 = random_deletion(transcript[split:], 
                                               min_d=sd_min, max_d=sd_max)
        else:
            transcript_half1 = transcript[:split]
            transcript_half2 = transcript[split:]
        transcript_half1 = " ".join(transcript_half1) + " "
        transcript_half2 = " ".join(transcript_half2)
        if model_type == 'bert':
            transcript_half2 = transcript_half2[:-5]
        elif model_type == 't5':
            transcript_half2 = transcript_half2[:-4] 

        if aug_vect[1] == 1: # apply back-translation
            transcript_half1 = aug_bt.augment(transcript_half1)
            transcript_half2 = aug_bt.augment(transcript_half2)

        transcript_aug = (transcript_half1, transcript_half2)

        return transcript_aug

    # now we cover cases where mixup is False
    
    if aug_vect[0] == 1: # apply deletion
        transcript = random_deletion(transcript, min_d=sd_min, max_d=sd_max*2)
        transcript_aug = " ".join(transcript)
        
        if model_type == 'bert':
            transcript_aug = transcript_aug[:-5]
        elif model_type == 't5':
            transcript_aug = transcript_aug[:-4] 
            
        if aug_vect[1] == 1:
            transcript_aug = aug_bt.augment(transcript_aug)
        
        return transcript_aug

    if aug_vect[1] == 1: # apply back-translation
        transcript = " ".join(transcript)
        if model_type == 'bert':
            transcript = transcript[:-5]
        elif model_type == 't5':
            transcript = transcript[:-4] 
        transcript_aug = aug_bt.augment(transcript)
    
    return transcript_aug