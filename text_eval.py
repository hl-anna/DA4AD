from sklearn.metrics import confusion_matrix
import torch
import numpy as np

def model_performance(output, target):
    """
    Returns accuracy per batch, 
    i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    cf = confusion_matrix(target, output, labels=[0,1])
    # print(cf)
    tp, fp, fn, tn = cf[1][1], cf[0][1], cf[1][0], cf[0][0]

    return tp, fp, fn, tn


def eval(data_iter, model, device):
    
    """
    Evaluating model performance on the dev set
    """
    
    model.eval()
    epoch_loss = 0
    epoch_correct = 0
    epoch_incorrect = 0
    tps = 0
    pred_all = []
    trg_all = []
    no_observations = 0

    with torch.no_grad():
        for batch in data_iter:
            ids, labels = batch
            ids, labels = ids.to(device), labels.to(device)
            out = model(ids, labels = labels)

            loss, preds = out[0], out[1]
            no_observations += labels.shape[0]
            
            tp, fp, fn, tn = model_performance(
                np.argmax(preds.detach().cpu().numpy(), axis=1), 
                labels.cpu().numpy())

            epoch_loss += loss.item()*labels.shape[0]
            tps += tp
            epoch_correct += (tp+tn)
            epoch_incorrect += (fn+fp)
            pred_all.extend(preds.detach())
            trg_all.extend(labels.detach())
            
    acc = epoch_correct / no_observations
    f1 = tps / (tps + 0.5*(epoch_incorrect))
    loss = epoch_loss / no_observations

    return loss, acc, f1