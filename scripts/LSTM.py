import numpy as np
import csv, gzip, os, sys, gc
import math
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F

import logging
import datetime
import optparse
import pandas as pd
import os
from sklearn.metrics import log_loss
import ast
from torch.utils.data import Dataset
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from scipy.ndimage import uniform_filter
from torch.optim.lr_scheduler import StepLR

from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
from utils import dumpobj, loadobj, GradualWarmupScheduler

package_dir = ''
sys.path.append(package_dir)
sys.path.insert(0, 'scripts')

# Print info about environments
print('Cuda set up : time {}'.format(datetime.datetime.now().time()))

device = torch.device('cuda')
print('Device : {}'.format(torch.cuda.get_device_name(0)))
print('Cuda available : {}'.format(torch.cuda.is_available()))

n_gpu = torch.cuda.device_count()
print('Cuda n_gpus : {}'.format(n_gpu))


SEED = 1234
IMG_SIZE = 480
GLOBALEPOCH = 3
n_epochs = 12 
lr = 0.00001
lrgamma = 0.95
DECAY = 0
batch_size = 4
ROOT = ''
path_data = os.path.join(ROOT, 'data')
path_img = os.path.join(ROOT, 'img')
WORK_DIR = os.path.join(ROOT, 'work')
path_emb = os.path.join(ROOT, 'work')
traindataset = ''
testdataset = 'CQ500'
testmetadata_name = 'test_metadata.csv' if testdataset=='' else f'{testdataset}_test_metadata.csv'
WEIGHTS_NAME = 'pytorch_model.bin'
fold = 6
LSTM_UNITS = 128
EMB_SIZE = 2048
nbags = 12
DROPOUT = 0.3
TTAHFLIP = 'F'
TTATRANSPOSE = 'F'
INFER = 'TST'   # 'TRN' or 'TST'
n_classes = 6
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']



################################################################################################################
################################################################################################################

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

def criterion(data, targets, criterion = torch.nn.BCEWithLogitsLoss()):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    loss_all = criterion(data, targets)
    loss_any = criterion(data[:,-1:], targets[:,-1:])
    return (loss_all*6 + loss_any*1)/7

class IntracranialDataset(Dataset):
    def __init__(self, df, mat, labels=label_cols):
        self.data = df
        self.mat = mat
        self.labels = labels
        # Patients = unique sliceIDs in the df
        self.patients = df.SliceID.unique()
        # Set the df to index by SliceID
        self.data = self.data.set_index('SliceID')

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):

        # Get patient idx based on unique sliceID
        patidx = self.patients[idx]
        # Sort all slices of patient into sequence
        patdf = self.data.loc[patidx].sort_values('seq')
        # Get actual embeddings of each slice for this patient
        patemb = self.mat[patdf['embidx'].values]

        # Concat on the deltas between current and prev/next embeddings (current-prev and current-next embedding)
        # This gives the model knowledge of changes around the image
        patdeltalag  = np.zeros(patemb.shape)
        patdeltalead = np.zeros(patemb.shape)
        patdeltalag [1:] = patemb[1:]-patemb[:-1]
        patdeltalead[:-1] = patemb[:-1]-patemb[1:]

        patemb = np.concatenate((patemb, patdeltalag, patdeltalead), -1)

        # Retrieve embidx of all embeddings attributed to this patient
        ids = torch.tensor(patdf['embidx'].values)

        if self.labels:
            labels = torch.tensor(patdf[label_cols].values)
            return {'emb': patemb, 'embidx' : ids, 'labels': labels}    
        else:      
            return {'emb': patemb, 'embidx' : ids}

class NeuralNet(nn.Module):
    def __init__(self, embed_size=trnemb.shape[-1]*3, LSTM_UNITS=64, DO = 0.3):
        super(NeuralNet, self).__init__()
        
        self.embedding_dropout = SpatialDropout(0.0) #DO)

        # nn.LSTM(input_size, hidden_size, num_layers=1)
        # input to LSTM/RNN is of shape: (batch_size, seq_length, input_size)
        # output of LSTM/RNN is of shape: (batch_size, seq_length, hidden_size)
        # note: * 2 is needed because it is bidirectional (i.e. produces x2 LSTMs)
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.linear = nn.Linear(LSTM_UNITS*2, n_classes)

    def forward(self, x, lengths=None):
        h_embedding = x

        h_embadd = torch.cat((h_embedding[:,:,:EMB_SIZE], h_embedding[:,:,:EMB_SIZE]), -1)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2))
        
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 + h_embadd

        output = self.linear(hidden)
        
        return output

# a simple custom collate function, just to show the idea
def collatefn(batch):
    # for each img in the batch, get the lengths of each emb and find the longest length
    # get emb dimension (same for each emb)
    # get label dimension if label present (same for each label)
    maxlen = max([l['emb'].shape[0] for l in batch])
    embdim = batch[0]['emb'].shape[1]
    withlabel = 'labels' in batch[0]
    if withlabel:
        labdim = batch[0]['labels'].shape[1]

    for b in batch:
        # make all sequences in the batch have the same length by padding
        # find the amount of padding needed
        masklen = maxlen - len(b['emb'])
        # add padding (padded with 0) to the embedding itself
        b['emb'] = np.vstack((np.zeros((masklen, embdim)), b['emb']))
        # add padding to the embidx list
        b['embidx'] = torch.cat((torch.ones((masklen), dtype=torch.long) * -1, b['embidx']))
        # set mask to 1 at padding parts; other parts incl. embedding is 0
        b['mask'] = np.ones((maxlen))
        b['mask'][:masklen] = 0.
        if withlabel:
            # add padding (padded with 0) to labels
            b['labels'] = np.vstack((np.zeros((maxlen - len(b['labels']), labdim)), b['labels']))

    # create outbatch, which contains the emb, mask, embidx, label of each item in the batch
    # outbatch['emb'] is shape(batch_size, shape(emb))
    outbatch = {'emb': torch.tensor(np.vstack([np.expand_dims(b['emb'], 0) \
                                               for b in batch])).float()}
    outbatch['mask'] = torch.tensor(np.vstack([np.expand_dims(b['mask'], 0) \
                                               for b in batch])).float()
    outbatch['embidx'] = torch.tensor(np.vstack([np.expand_dims(b['embidx'], 0) \
                                                 for b in batch])).float()
    if withlabel:
        outbatch['labels'] = torch.tensor(np.vstack([np.expand_dims(b['labels'], 0) for b in batch])).float()
    return outbatch

def loademb(TYPE, dataset_name, IMG_SIZE, fold, GLOBALEPOCH, TTA=''):
    return np.load(os.path.join(path_emb,
                                'emb{}_{}{}_size{}_fold{}_ep{}.npz'.format(TTA, TYPE, dataset_name, IMG_SIZE, fold,
                                                                            GLOBALEPOCH)))['arr_0']

def makeSub(ypred, imgs):
    imgls = np.array(imgs).repeat(len(label_cols))
    icdls = pd.Series(label_cols*ypred.shape[0])
    yidx = ['{}_{}'.format(i,j) for i,j in zip(imgls, icdls)]
    subdf = pd.DataFrame({'ID': yidx, 'Label': ypred.flatten()})
    return subdf

def predict(loader):
    valls = []
    imgls = []
    imgdf = loader.dataset.data.reset_index().set_index('embidx')[['Image']].copy()
    for step, batch in enumerate(loader):
        inputs = batch["emb"]
        mask = batch['mask'].to(device, dtype=torch.int)
        inputs = inputs.to(device, dtype=torch.float)
        logits = model(inputs)
        # get the mask for masked labels
        maskidx = mask.view(-1)==1
        # reshape for
        logits = logits.view(-1, n_classes)[maskidx]
        valls.append(torch.sigmoid(logits).detach().cpu().numpy())
        # Get the list of images
        embidx = batch["embidx"].detach().cpu().numpy().astype(np.int32)
        embidx = embidx.flatten()[embidx.flatten()>-1]
        images = imgdf.loc[embidx].Image.tolist()
        imgls += images
    return np.concatenate(valls, 0), imgls


################################################################################################################
################################################################################################################

########### Sort slices into sequence using metadata ###########

# Print info about environments
print('Cuda set up : time {}'.format(datetime.datetime.now().time()))

# Get image sequences
trnmdf = pd.read_csv(os.path.join(path_data, 'train_metadata.csv'))
tstmdf = pd.read_csv(os.path.join(path_data, testmetadata_name))
for col in tstmdf.columns:
    if col == 'CT-idx_sliceidx':
        tstmdf.rename(columns={'SOPInstanceUID': 'SOPInstanceUID_old', 'CT-idx_sliceidx': 'SOPInstanceUID'}, inplace=True)
# tstmdf.columns = ['SOPInstanceUID' if x=='CT-idx_sliceidx' else x for x in tstmdf.columns]

# Create a SliceID which has format: PatientID_SeriesInstanceUID_StudyInstanceUID
trnmdf['SliceID'] = trnmdf[['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID']].apply(lambda x: '{}__{}__{}'.format(*x.tolist()), 1)
tstmdf['SliceID'] = tstmdf[['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID']].apply(lambda x: '{}__{}__{}'.format(*x.tolist()), 1)

# Each ImagePositionPatient is originally a list of 3 float numbers [a,b,c]
# Change this to separate columns of: ImagePos1, ImagePos2, ImagePos3
poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
tstmdf[poscols] = pd.DataFrame(tstmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())

# Sort by SliceID and ImagePos, then extract just the created SliceID, PatientID and SOPInstanceUID (same as .png ID)
trnmdf = trnmdf.sort_values(['SliceID']+poscols)\
                [['PatientID', 'SliceID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
tstmdf = tstmdf.sort_values(['SliceID']+poscols)\
                [['PatientID', 'SliceID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
# tstmdf = tstmdf.sort_values(['SliceID']+poscols)\
#                 [['PatientID', 'SliceID', 'CT-idx_sliceidx']+poscols].reset_index(drop=True)

# Group by SliceID (now in seq), then index it by the cumulative count +1 (i.e. first slice = 1, second slice = 2)
trnmdf['seq'] = (trnmdf.groupby(['SliceID']).cumcount() + 1)
tstmdf['seq'] = (tstmdf.groupby(['SliceID']).cumcount() + 1)

# Entire dataframe now only has these 4 columns
trnkeepcols = ['PatientID', 'SliceID', 'SOPInstanceUID', 'seq']
tstkeepcols = ['PatientID', 'SliceID', 'SOPInstanceUID', 'seq']
# tstkeepcols = ['PatientID', 'SliceID', 'CT-idx_sliceidx', 'seq']
trnmdf = trnmdf[trnkeepcols]
tstmdf = tstmdf[tstkeepcols]

# Rename the columns
trnmdf.columns = tstmdf.columns = ['PatientID', 'SliceID', 'Image', 'seq']

# Load Data Frames
trndf = loadobj(os.path.join(path_emb, 'loader{}_trn{}_size{}_fold{}_ep{}'.format(
    TTAHFLIP + TTATRANSPOSE, traindataset, IMG_SIZE, fold, GLOBALEPOCH))).dataset.data
valdf = loadobj(os.path.join(path_emb, 'loader{}_val{}_size{}_fold{}_ep{}'.format(
    TTAHFLIP + TTATRANSPOSE, traindataset, IMG_SIZE, fold, GLOBALEPOCH))).dataset.data
tstdf = loadobj(os.path.join(path_emb, 'loader{}_tst{}_size{}_fold{}_ep{}'.format(
    TTAHFLIP + TTATRANSPOSE, testdataset, IMG_SIZE, fold, GLOBALEPOCH))).dataset.data

# Create emb idx for each row in the df (contains images fr prev used dataloader in CNN training)
# i.e. Each slice has a corresponding embidx, indexed in the same order as they were processed from the dataloader
trndf['embidx'] = range(trndf.shape[0])
valdf['embidx'] = range(valdf.shape[0])
tstdf['embidx'] = range(tstdf.shape[0])

# Merge trndf (contains images fr prev used dataloader) with trnmdf (metadata) based on 'Image' column
# trndf is now sorted in sequence and contains embidx
trndf = trndf.merge(trnmdf.drop('PatientID', 1), on = 'Image')
valdf = valdf.merge(trnmdf.drop('PatientID', 1), on = 'Image')
tstdf = tstdf.merge(tstmdf, on = 'Image')

print('Trn df shape {} {}'.format(*trndf.shape))
print('Val df shape {} {}'.format(*valdf.shape))
print('Tst df shape {} {}'.format(*tstdf.shape))


########### Load embeddings ###########

# Load embeddings
embnm = 'emb_sz256_wt256_fold{}_epoch{}'.format(fold, GLOBALEPOCH)
print('Load npy..')

print('Load embeddings...')
trnembls = [loademb('trn', traindataset, IMG_SIZE, fold, GLOBALEPOCH)]
valembls = [loademb('val', traindataset, IMG_SIZE, fold, GLOBALEPOCH)]
tstembls = [loademb('tst', testdataset, IMG_SIZE, fold, GLOBALEPOCH)]

if TTAHFLIP == 'T':
    print('Load hflip...')
    trnembls.append(loademb('trn', traindataset, IMG_SIZE, fold, GLOBALEPOCH, TTA='T'))
    valembls.append(loademb('val', traindataset, IMG_SIZE, fold, GLOBALEPOCH, TTA='T'))
    tstembls.append(loademb('tst', testdataset, IMG_SIZE, fold, GLOBALEPOCH, TTA='T'))
if TTATRANSPOSE == 'P':
    print('Load transpose...')
    trnembls.append(loademb('trn', traindataset, IMG_SIZE, fold, GLOBALEPOCH, TTA='P'))
    valembls.append(loademb('val', traindataset, IMG_SIZE, fold, GLOBALEPOCH, TTA='P'))
    tstembls.append(loademb('tst', testdataset, IMG_SIZE, fold, GLOBALEPOCH, TTA='P'))

trnemb = sum(trnembls) / len(trnembls)
valemb = sum(valembls) / len(valembls)
tstemb = sum(tstembls) / len(tstembls)

print('Trn shape {} {}'.format(*trnemb.shape))
print('Val shape {} {}'.format(*valemb.shape))
print('Tst shape {} {}'.format(*tstemb.shape))
print('Add stg1 test labels to train')
del trnembls, valembls, tstembls
gc.collect()


########### Create dataloaders ###########

print('Create loaders...')
trndataset = IntracranialDataset(trndf, trnemb, labels=True)
trnloader = DataLoader(trndataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collatefn)

valdataset = IntracranialDataset(valdf, valemb, labels=False)
tstdataset = IntracranialDataset(tstdf, tstemb, labels=False)

tstloader = DataLoader(tstdataset, batch_size=batch_size * 4, shuffle=False, num_workers=8, collate_fn=collatefn)
valloader = DataLoader(valdataset, batch_size=batch_size * 4, shuffle=False, num_workers=8, collate_fn=collatefn)



################################################################################################################
################################################################################################################

print('Create model')
model = NeuralNet(LSTM_UNITS=LSTM_UNITS, DO=DROPOUT)
model = model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
plist = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': DECAY},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = optim.Adam(plist, lr=lr)
scheduler = StepLR(optimizer, 1, gamma=lrgamma, last_epoch=-1)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


if INFER == 'TRN':
    
    ypredtstls = []
    
    for gepoch in range(GLOBALEPOCH):
        print('Global epoch: {}'.format(gepoch))
        
        for epoch in range(n_epochs):
                tr_loss = 0.
                for param in model.parameters():
                    param.requires_grad = True
                model.train()
                for step, batch in enumerate(trnloader):
                    y = batch['labels'].to(device, dtype=torch.float)
                    mask = batch['mask'].to(device, dtype=torch.int)
                    x = batch['emb'].to(device, dtype=torch.float)
                    x = torch.autograd.Variable(x, requires_grad=True)
                    y = torch.autograd.Variable(y)
                    logits = model(x).to(device, dtype=torch.float)
                    # get the mask for masked labels
                    maskidx = mask.view(-1)==1
                    y = y.view(-1, n_classes)[maskidx]
                    logits = logits.view(-1, n_classes)[maskidx]
                    # Get loss
                    loss = criterion(logits, y)

                    tr_loss += loss.item()
                    optimizer.zero_grad()
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()
                    if step%1000==0:
                        print('Trn step {} of {} trn lossavg {:.5f}'. \
                                    format(step, len(trnloader), (tr_loss/(1+step))))
                output_model_file = os.path.join(WORK_DIR, 'weights/lstm_gepoch{}_lstmepoch{}_fold{}.bin'.format(GLOBALEPOCH, epoch, fold))
                torch.save(model.state_dict(), output_model_file)

                scheduler.step()
                model.eval()

                ypred, imgval = predict(valloader)
                yvalp = makeSub(ypred, imgval)
                
                ypredls.append(ypred)                
                yvalpred = sum(ypredls[-nbags:])/len(ypredls[-nbags:])
                yvalout = makeSub(yvalpred, imgval)

                weights = ([1, 1, 1, 1, 1, 2] * ypred.shape[0])
                yact = valloader.dataset.data[label_cols].values#.flatten()
                yact = makeSub(yact, valloader.dataset.data['Image'].tolist())
                yact = yact.set_index('ID').loc[yvalout.ID].reset_index()
                valloss = log_loss(yact['Label'].values, yvalp['Label'].values.clip(.00001,.99999) , sample_weight = weights)
                vallossavg = log_loss(yact['Label'].values, yvalout['Label'].values.clip(.00001,.99999) , sample_weight = weights)
                print('Epoch {} val logloss {:.5f} bagged {:.5f}'.format(epoch, valloss, vallossavg))

                ypred, imgtst = predict(tstloader)
                ypredtstls.append(ypred)

            
if INFER == 'TST':
    # Default is to test using the last trained lstm epoch of the last global epoch #
    gepoch = GLOBALEPOCH
    del model
    print('Load model')
    model = NeuralNet(LSTM_UNITS=LSTM_UNITS, DO=DROPOUT)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    input_model_file = os.path.join(WORK_DIR, 'weights/lstm_gepoch{}_lstmepoch{}_fold{}.bin'.format(gepoch, epoch, fold))
    model.load_state_dict(torch.load(input_model_file))
    model = model.to(device)

    model.eval()
    print('Making predictions...')
    ypred, imgtst = predict(tstloader)

    ytstout = makeSub(ypred, imgtst)
    ytstout.to_csv('preds/{}lstm{}{}{}_{}_epoch{}_sub_{}.csv.gz'.format(testdataset, TTAHFLIP, TTATRANSPOSE, LSTM_UNITS, WORK_DIR.split('/')[-1], epoch, embnm), \
                index = False, compression = 'gzip')
