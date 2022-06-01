from setup import torch_setup, set_seed
import sys
sys.path.append('./ast')
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from src.models import ASTModel

import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
print('ADRESSO DATASET')


class Dataset_AST(Dataset):

    def __init__(self, aug_type):
        self.data_dir = "./data_" + aug_type
        self.data_list = os.listdir(self.data_dir)
        data_list_int = np.array([int(d[:-4]) for d in self.data_list])
        self.data_list = np.array(self.data_list)
        self.data_list = self.data_list[np.argsort(data_list_int)]
        if aug_type=="test":
            self.label_dir = "./test_segment_labels"
        else:
            self.label_dir = "./train_labels_" + aug_type
        self.labels = pd.read_csv(self.label_dir, header=None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data = pd.read_csv(os.path.join(self.data_dir, self.data_list[item]), header=None).to_numpy()
        data = torch.tensor(data, dtype=torch.float)
        label = torch.tensor(self.labels.iloc[item, 0], dtype=torch.long)
        return data, label



def model_performance(output, target, print_output=False):
    """
    Returns accuracy per batch, 
    i.e. if you get 8/10 right, this returns 0.8
    """
    correct_answers = (output == target)
    correct = sum(correct_answers)
    acc = np.true_divide(correct,len(output))

    if print_output:
        print(f'| Acc: {acc:.2f} ')

    return correct, acc

def eval(data_iter, model, criterion):
    """
    Evaluating model performance on the dev set
    """
    model.eval()
    epoch_loss = 0
    epoch_correct = 0
    pred_all = []
    trg_all = []
    no_observations = 0

    with torch.no_grad():
        for batch in data_iter:
            id, labels = batch
            id = id.to(DEVICE)
            out = model(id)
            loss = criterion(out, labels)
            no_observations += labels.shape[0]
            
            correct, __ = model_performance(
                np.argmax(out.detach().cpu().numpy(), axis=1), 
                labels.cpu().numpy())

            epoch_loss += loss.item()*labels.shape[0]
            epoch_correct += correct

    return epoch_loss/no_observations, epoch_correct/no_observations


def train(train_iter, model, number_epoch, optimizer, criterion, scheduler=None, dev_iter = None):
    """
    Training loop for the model, which calls on eval to evaluate after each epoch
    """
    print("Training model.")
    model = model.to(DEVICE)
    epoch_info = []

    for epoch in range(1, number_epoch+1):
        
        model.train()
        
        epoch_loss = 0
        epoch_correct = 0
        no_observations = 0  # Observations used for training so far
        print('epoch number: ', epoch)
        for batch in tqdm(train_iter):
            ids, labels = batch
            ids, labels = ids.to(DEVICE), labels.to(DEVICE)
            no_observations = no_observations + labels.shape[0]

            optimizer.zero_grad()
            out = model(ids)
            ids.detach().cpu().numpy()
            del ids
            correct, __ = model_performance(
                np.argmax(out.detach().cpu().numpy(), axis=1), 
                labels.cpu().numpy())
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()*labels.shape[0]
            epoch_correct += correct
            del labels

        if dev_iter:
            valid_loss, valid_acc = eval(dev_iter, model, criterion)

        epoch_loss, epoch_acc = epoch_loss / no_observations, epoch_correct / no_observations 
        if dev_iter:       
            print(f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.2f} | Train Accuracy: {epoch_acc:.2f} | \
            Val. Loss: {valid_loss:.2f} | Val. Accuracy: {valid_acc:.2f} |')
        else:            
            print(f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.2f} | Train Accuracy: {epoch_acc:.2f} |')
            epoch_info.append([epoch_loss, epoch_acc])



test_sample_labels = pd.read_csv("./test_labels.txt", sep='\t', header=None)
test_sample_labels = test_sample_labels.iloc[:,1]
test_seg_id_a = pd.read_csv("test_seg_id_a.csv")
test_seg_id_a = test_seg_id_a.iloc[:,1]
def segment_to_test(y_pred, y_pred_proba, test_seg_id_a, test_sample_labels):
    y_pred_sample = []
    pred_labels = []
    test_ids = np.unique(test_seg_id_a)
    for id in test_ids:
        id_pred_proba = np.mean(y_pred_proba[test_seg_id_a==id], axis=0)
        y_pred_sample.append([id, id_pred_proba[0], id_pred_proba[1]])
        
        id_preds = y_pred[test_seg_id_a==id]
        ones = sum(id_preds==1)
        zeros = sum(id_preds==0)
        if ones>=zeros:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
            
        
    f1 = f1_score(test_sample_labels, pred_labels)
    acc = sum(test_sample_labels==pred_labels)/48
    
    return acc, f1, y_pred_sample

DEVICE = torch_setup()
print(DEVICE)
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
use_autovc = False
use_augment = True
augmentation_type = 'loudness' #["masking", "shift", "loudness", "noise", "pitch", "vtlp", "shift", "speed", "normalisation", "specaugment"]
random_aug_size = 2

print('AUGMENTATION TYPE IS:  ', augmentation_type)

acc_results = []
for i in range(1,6):
    SEED = set_seed(i)
    train_dataset = Dataset_AST(augmentation_type)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #, shuffle=True

    model = ASTModel(label_dim=2, fstride=16, tstride=16, input_fdim=128, input_tdim=1601, 
            imagenet_pretrain=True, audioset_pretrain=False, model_size='base384')

    
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-06)
    criterion = CrossEntropyLoss()
    train(train_dataloader, model, EPOCHS, optimizer, criterion)
    test_dataset = Dataset_AST("test")
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    pred_list = []
    pred_sig_list = []
    breaker = 0
    for batch in test_dataloader:
        ids, labels = batch
        ids, labels = ids.to(DEVICE), labels.to(DEVICE)
        out = model(ids)

        pred = np.argmax(out.detach().cpu().numpy())
        pred_sig = np.array(out.detach().cpu().numpy())
        pred_list.append(pred)
        pred_sig_list.append(pred_sig[0])
        breaker += 1

    pred_list = np.array(pred_list)
    pred_sig_list = np.array(pred_sig_list)
    score_acc, score_f1, y_pred_sample = segment_to_test(pred_list, pred_sig_list, test_seg_id_a, test_sample_labels)
    acc_results.append([score_acc, score_f1])
    pd.DataFrame(y_pred_sample).to_csv("./ast_test_preds/" + augmentation_type + '_' + str(i) + '.csv')

pd.DataFrame(acc_results).to_csv("./ast_test_preds/" + augmentation_type + '_acc.csv')
