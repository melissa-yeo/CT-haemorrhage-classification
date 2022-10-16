import os, sys
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

import cv2
import glob
import cv2
import gc
import datetime

import numpy as np
import pandas as pd
import torch.optim as optim

from torch.utils.data import Dataset
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck
from apex import amp

from albumentations.pytorch import ToTensor
from albumentations import (Cutout, Compose, Normalize, RandomRotate90, HorizontalFlip,
                            VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,
                            GaussNoise, RandomGamma, RandomContrast, RandomBrightness, HueSaturationValue,
                            RandomBrightnessContrast, Lambda, NoOp, CenterCrop, Resize
                            )

from tqdm import tqdm
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
EPOCHS = 3
AUTOCROP = 'T'
lr = float(0.00002)
batch_size = int(32)

ROOT = ''
path_data = os.path.join(ROOT, 'data')
testcsvgz_name = ''
dataset_name = 'CQ500'
path_img = os.path.join(ROOT, '')
WORK_DIR = os.path.join(ROOT, '')

fold = 5

INFER = ''  # TRN or EMB or TST
HFLIP = 'F'
TRANSPOSE = 'F'

n_epochs = EPOCHS
device = 'cuda'
print('Data path : {}'.format(path_data))
print('Image path : {}'.format(path_img))

os.environ["TORCH_HOME"] = os.path.join(path_data, 'mount')



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold
    Crops blank image to 1x1.
    Returns cropped image.
    https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    """

    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    cols = np.where(np.max(flatImage, 1) > threshold)[0]
    image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    # print(image.shape)
    sqside = max(image.shape)
    imageout = np.zeros((sqside, sqside, 3), dtype='uint8')
    imageout[:image.shape[0], :image.shape[1], :] = image.copy()
    return imageout


class IntracranialDataset(Dataset):

    def __init__(self, df, path, labels, imglist=False, transform=None):
        self.path = path
        self.data = df
        self.transform = transform
        self.labels = labels
        self.imglist = imglist
        self.crop = AUTOCROP

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, 'Image']
        img_path = os.path.join(self.path, img_name + '.jpg')
        img = cv2.imread(img_path)
        if self.crop:
            try:
                img = autocrop(img, threshold=0)
            except:
                1
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        if self.labels:
            labels = torch.tensor(self.data.loc[idx, label_cols])
            return {'image': img, 'labels': labels}
        if self.imglist:
            sub_idx = torch.tensor(int(img_name.split('_')[0].split('-')[1]))
            slc_idx = torch.tensor(int(img_name.split('_')[1][2:]))
            return {'image': img, 'sub_idx': sub_idx, 'slc_idx': slc_idx}
        else:
            return {'image': img}


np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

print('Load Dataframes')
dir_train_img = os.path.join(path_img)
dir_test_img = os.path.join(path_img)

# Parameters
n_classes = 6
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

train = pd.read_csv(os.path.join(path_data, 'train.csv.gz'))
test = pd.read_csv(os.path.join(path_data, testcsvgz_name))

png = glob.glob(os.path.join(dir_train_img, '*.jpg'))
png = [os.path.basename(png)[:-4] for png in png]

train_imgs = set(train.Image.tolist())
png = [p for p in png if p in train_imgs]
print('Number of images to train on {}'.format(len(png)))
png = np.array(png)
train = train.set_index('Image').loc[png].reset_index()


# Training and validation dfs
valdf = train[train['fold'] == fold].reset_index(drop=True)
trndf = train[train['fold'] != fold].reset_index(drop=True)
print('Trn set shape {} {}'.format(*trndf.shape))
print('Val set shape {} {}'.format(*valdf.shape))


# Data loaders
mean_img = [0.22363983, 0.18190407, 0.2523437]
std_img = [0.32451536, 0.2956294, 0.31335256]
transform_train = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                     rotate_limit=20, p=0.3, border_mode=cv2.BORDER_REPLICATE),
    Transpose(p=0.5),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
])

HFLIPVAL = 1.0 if HFLIP == 'T' else 0.0
TRANSPOSEVAL = 1.0 if TRANSPOSE == 'P' else 0.0
transform_test = Compose([
    HorizontalFlip(p=HFLIPVAL),
    Transpose(p=TRANSPOSEVAL),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
])

trndataset = IntracranialDataset(trndf, path=dir_train_img, transform=transform_train, labels=True)
valdataset = IntracranialDataset(valdf, path=dir_train_img, transform=transform_test, labels=False)
tstdataset = IntracranialDataset(test, path=dir_test_img, transform=transform_test, labels=False)


num_workers = 0
trnloader = DataLoader(trndataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valloader = DataLoader(valdataset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)
tstloader = DataLoader(tstdataset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

model = torch.load('checkpoints/resnext101_32x8d_wsl_checkpoint.pth')
model.fc = torch.nn.Linear(2048, n_classes)
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()

# def criterion(data, targets, criterion=torch.nn.BCEWithLogitsLoss()):
#     ''' Define custom loss function for weighted BCE on 'target' column '''
#     loss_all = criterion(data, targets)
#     loss_any = criterion(data[:, -1:], targets[:, -1:])
#     return (loss_all * 6 + loss_any * 1) / 7
# def criterion(data, targets):
#     ''' Define custom loss function for weighted BCE on 'target' column, increase weight for EDH '''
#     pos_weight = torch.FloatTensor([2,1,1,1,1,1]).cuda()
#     criterionBCE = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#     loss_all = criterionBCE(data, targets)
#     loss_any = criterionBCE(data[:, -1:], targets[:, -1:])
#     return (loss_all * 6 + loss_any * 1) / 7
def criterion(data, targets):
    ''' Define custom loss function using focal loss, on 'target' column '''
    criterionFL = kornia.losses.FocalLoss(alpha=0.5, gamma=2.0, reduction='none')
    loss_all = criterionFL(data, targets)
    loss_any = criterionFL(data[:, -1:], targets[:, -1:])
    return (loss_all * 6 + loss_any * 1) / 7


plist = [{'params': model.parameters(), 'lr': lr}]
optimizer = optim.Adam(plist, lr=lr)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

val_loss_min = np.Inf

for epoch in range(n_epochs):
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)
    
    if INFER == 'TRN':
        for param in model.parameters():
            param.requires_grad = True
            
        # Train CNN #
        model.train()
        tr_loss = 0
        for step, batch in enumerate(trnloader):
            if step % 1000 == 0:
                print('Train step {} of {}'.format(step, len(trnloader)))
            optimizer.zero_grad()
            inputs = batch["image"]
            labels = batch["labels"]
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            del inputs, labels, outputs
            
        # Validate CNN # 
        model.eval()
        val_loss = 0
        for step, batch in enumerate(valloader):
            if step % 1000 == 0:
                print('Train step {} of {}'.format(step, len(trnloader)))
            inputs = batch["image"]
            labels = batch["labels"]
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            del inputs, labels, outputs
         
        epoch_tr_loss = tr_loss / len(trnloader)
        epoch_val_loss = val_loss / len(valloader)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch+1, epoch_tr_loss, epoch_val_loss))
        
        if epoch_val_loss <= val_loss_min:
            for param in model.parameters():
                param.requires_grad = False
            output_model_file = os.path.join(WORK_DIR, 'weights/model_epoch{}_fold{}.bin'.format(epoch, fold))
            torch.save(model.state_dict(), output_model_file)
            val_loss_min = epoch_val_loss

            
    if INFER == 'EMB' or INFER == 'TST' or INFER == 'TST_NOEMB':
        # Load trained CNN # 
        del model
        # model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
        model = torch.load('checkpoints/resnext101_32x8d_wsl_checkpoint.pth')
        model.fc = torch.nn.Linear(2048, n_classes)
        device = torch.device("cuda:{}".format(n_gpu - 1))
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)[::-1]), output_device=device)
        for param in model.parameters():
            param.requires_grad = False
        input_model_file = os.path.join(WORK_DIR, 'weights/model_epoch{}_fold{}.bin'.format(epoch, fold))
        model.load_state_dict(torch.load(input_model_file))
        model = model.to(device)
        model.eval()
        print(model.parameters())

    if INFER == 'EMB':
        # For LSTM training #
        print('Output embeddings epoch {}'.format(epoch))
        print('Train shape {} {}'.format(*trndf.shape))
        print('Valid shape {} {}'.format(*valdf.shape))
        print('Test  shape {} {}'.format(*test.shape))
        trndataset = IntracranialDataset(trndf, path=dir_train_img, transform=transform_test, labels=False)
        valdataset = IntracranialDataset(valdf, path=dir_train_img, transform=transform_test, labels=False)
        tstdataset = IntracranialDataset(test, path=dir_test_img, transform=transform_test, labels=False)

        trnloader = DataLoader(trndataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        valloader = DataLoader(valdataset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)
        tstloader = DataLoader(tstdataset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

        # Extract embedding layer
        model.module.fc = Identity()
        model.eval()
        DATASETS = ['tst', 'val', 'trn']
        LOADERS = [tstloader, valloader, trnloader]
        for typ, loader in zip(DATASETS, LOADERS):
            ls = []
            for step, batch in enumerate(loader):
                if step % 1000 == 0:
                    print('Embedding {} step {} of {}'.format(typ, step, len(loader)))
                inputs = batch["image"]
                inputs = inputs.to(device, dtype=torch.float)
                out = model(inputs)
                ls.append(out.detach().cpu().numpy())
            outemb = np.concatenate(ls, 0).astype(np.float32)
            print('Write embeddings : shape {} {}'.format(*outemb.shape))
            fembname = 'emb{}_{}_size{}_fold{}_ep{}'.format(HFLIP + TRANSPOSE, typ, IMG_SIZE, fold, epoch)
            print('Embedding file name : {}'.format(fembname))
            np.savez_compressed(
                os.path.join(WORK_DIR, 'emb{}_{}_size{}_fold{}_ep{}'.format(HFLIP + TRANSPOSE, typ, IMG_SIZE, fold, epoch)),
                outemb)
            dumpobj(os.path.join(WORK_DIR,
                                 'loader{}_{}_size{}_fold{}_ep{}'.format(HFLIP + TRANSPOSE, typ, IMG_SIZE, fold, epoch)),
                    loader)
            gc.collect()

    if INFER == 'TST':
        # For LSTM testing #
        print('Output embeddings epoch {}'.format(epoch))
        print('Test  shape {} {}'.format(*test.shape))
        tstdataset = IntracranialDataset(test, path=dir_test_img, transform=transform_test, labels=False)
        tstloader = DataLoader(tstdataset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

        # Extract embedding layer
        model.module.fc = Identity()
        model.eval()
        DATASETS = ['tst']
        LOADERS = [tstloader]
        for typ, loader in zip(DATASETS, LOADERS):
            ls = []
            for step, batch in enumerate(loader):
                if step % 1000 == 0:
                    print('Embedding {} step {} of {}'.format(typ, step, len(loader)))
                inputs = batch["image"]
                inputs = inputs.to(device, dtype=torch.float)
                out = model(inputs)
                ls.append(out.detach().cpu().numpy())
            outemb = np.concatenate(ls, 0).astype(np.float32)
            print('Write embeddings : shape {} {}'.format(*outemb.shape))
            fembname = 'emb{}_{}{}_size{}_fold{}_ep{}'.format(HFLIP + TRANSPOSE, typ, dataset_name, IMG_SIZE, fold, epoch)
            print('Embedding file name : {}'.format(fembname))
            np.savez_compressed(os.path.join(WORK_DIR, 'emb{}_{}{}_size{}_fold{}_ep{}'.format(HFLIP + TRANSPOSE, typ,
                                                                                              dataset_name, IMG_SIZE, fold,
                                                                                              epoch)), outemb)
            dumpobj(os.path.join(WORK_DIR,
                                 'loader{}_{}{}_size{}_fold{}_ep{}'.format(HFLIP + TRANSPOSE, typ, dataset_name, IMG_SIZE,
                                                                           fold, epoch)), loader)
            gc.collect()
    
    
    if INFER == 'TST_NOEMB':
            # For CNN (no LSTM) testing #
            print('Testing with epoch {}'.format(epoch))
            print('Test  shape {} {}'.format(*test.shape))
            tstdataset = IntracranialDataset(test, path=dir_test_img, transform=transform_test, labels=False, imglist=True)
            tstloader = DataLoader(tstdataset, batch_size=batch_size*4, shuffle=False, num_workers=num_workers)
            # valdataset = IntracranialDataset(valdf, path=dir_train_img, transform=transform_test, labels=False)
            # valloader = DataLoader(valdataset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

            DATASETS = ['tst']
            LOADERS = [tstloader]
            for typ, loader in zip(DATASETS, LOADERS):
                valls = []
                subidxls = []
                slcidxls = []
                for step, batch in enumerate(loader):
                    if step%1000==0:
                        print('Test {} step {} of {}'.format(typ, step, len(loader)))
                    inputs = batch["image"].to(device, dtype=torch.float)
                    logits = model(inputs)
                    logits = logits.view(-1, n_classes)
                    valls.append(torch.sigmoid(logits).detach().cpu().numpy())
                    subidxls.append(batch["sub_idx"].detach().numpy())
                    slcidxls.append(batch["slc_idx"].detach().numpy())
                valls = np.concatenate(valls, 0)
                subidxls = np.concatenate(subidxls, 0)
                slcidxls = np.concatenate(slcidxls, 0)

            try:
                print('valls  shape: {}'.format(valls.shape))
                print('subidxls  shape: {}'.format(subidxls.shape))
                print('slcidxls  shape: {}'.format(slcidxls.shape))
            except:
                print('Could not print shape sizes :(')

            imgls = [f'CT-{i}_CT{j:06d}' for i,j in zip(subidxls, slcidxls)]
            print('Write out prediction to preds folder')
            ytstout = makeSub(valls, imgls)
            ytstout.to_csv('preds/{}CNN_{}_epoch{}.csv.gz'.format(dataset_name, WORK_DIR.split('/')[-1], epoch), index=False, compression='gzip')
