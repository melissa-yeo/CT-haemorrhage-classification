"""
1. Load 10 slices predicted to contain haemorrhage.
2. Run model to generate Grad-CAM images.
"""

import os, sys, glob
import optparse
import pandas as pd
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations import (Cutout, Compose, Normalize, RandomRotate90, HorizontalFlip,
                           VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,
                           GaussNoise, RandomGamma, RandomContrast, RandomBrightness, HueSaturationValue,
                           RandomBrightnessContrast, Lambda, NoOp, CenterCrop, Resize
                           )
from albumentations.pytorch import ToTensorV2

from gradcam_utils_v2 import *

#############################################################################
# Load variables and relevant files
#############################################################################

parser = optparse.OptionParser()
parser.add_option('-g', '--logmsg', action="store", dest="logmsg", help="root directory", default="")
parser.add_option('-k', '--dataset_name', action="store", dest="dataset_name", help="dataset name", default="")
parser.add_option('-r', '--rootpath', action="store", dest="rootpath", help="root directory", default="")
parser.add_option('-w', '--workpath', action="store", dest="workpath", help="Working path", default="")
parser.add_option('-p', '--predfilename', action="store", dest="predfilename", help="filename of preds.csv file", default="CQ500_scores.csv")
parser.add_option('-i', '--imageidx_filename', action="store", dest="imageidx_filename", help="filename of .csv file containing image idxs", default=None)
parser.add_option('-v', '--target_class_idx', action="store", dest="target_class_idx", help="target class to visualise", default=5)
parser.add_option('-c', '--size', action="store", dest="size", help="model size", default="512")
parser.add_option('-z', '--wtsize', action="store", dest="wtsize", help="model size", default="999")
parser.add_option('-e', '--epoch', action="store", dest="epoch", help="epoch", default="5")
parser.add_option('-o', '--fold', action="store", dest="fold", help="Fold for split", default="0")
parser.add_option('-y', '--autocrop', action="store", dest="autocrop", help="Autocrop", default="T")
parser.add_option('-b', '--batchsize', action="store", dest="batchsize", help="batch size", default="1")
options, args = parser.parse_args()

package_dir = options.rootpath
sys.path.append(package_dir)
sys.path.insert(0, 'scripts')

ROOT = options.rootpath
WORK_DIR = os.path.join(ROOT, options.workpath)
path_preds = os.path.join(ROOT, 'preds/')
path_data = os.path.join(ROOT, 'data/')
dataset_name = options.dataset_name
preds_name = options.predfilename
imageidx_filename = options.imageidx_filename
target_class_idx = int(options.target_class_idx)

SIZE = int(options.size)
WTSIZE = int(options.wtsize) if int(options.wtsize) != 999 else SIZE
epoch = int(options.epoch)
fold = int(options.fold)
AUTOCROP=options.autocrop=='T'
batch_size = int(options.batchsize)

n_classes = 6
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']


# Print info about environments
device=torch.device('cuda')
print('Device : {}'.format(torch.cuda.get_device_name(0)))
print('Cuda available : {}'.format(torch.cuda.is_available()))
n_gpu = torch.cuda.device_count()
print('Cuda n_gpus : {}'.format(n_gpu ))


#############################################################################
# Classes and functions
#############################################################################
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
    #print(image.shape)
    sqside = max(image.shape)
    imageout = np.zeros((sqside, sqside, 3), dtype = 'uint8')
    imageout[:image.shape[0], :image.shape[1],:] = image.copy()
    return imageout

class IntracranialDataset(Dataset):

    def __init__(self, df, path, labels, transform=None):
        self.path = path
        self.data = df
        self.transform = transform
        self.labels = labels
        self.crop = AUTOCROP

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.jpg')
        #img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_name)
        img_name = str(img_name.split('/')[-1][:-4])
        if self.crop:
            try:
                try:
                    img = autocrop(img, threshold=0, kernsel_size = image.shape[0]//15)
                except:
                    img = autocrop(img, threshold=0)
            except:
                1
        img = cv2.resize(img,(SIZE,SIZE))
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        if self.labels:
            labels = torch.tensor(
                self.data.loc[idx, label_cols])
            return {'image': img, 'labels': labels, 'img_name': img_name}
        else:
            return {'image': img, 'img_name': img_name}

class SaveFeatures():
    features=None
    def __init__(self, m):
        # attach the hook to the specified layer
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        # copy the activation features as an instance variable
        # stores the forward pass outputs (activation maps)
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()

def getCAM(feature_conv, weight_fc, class_idx):
    '''
    Index into the fc layer to get the weights for the class we want to investigate.
    Then calculate the dot product with our features from the image.

    :param feature_conv: activated features of the convnet
    :param weight_fc: weights of fc layer
    :param class_idx: class index we want to investigate (e.g. 283/'persian cat')
    :return:
    '''
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

  

#############################################################################
# Grad-CAM process (loveunk's fork)
#     1. Loads an image with opencv.
#     2. Preprocesses it for VGG19 and converts to a pytorch variable.
#     3. Makes a forward pass to find the category index with the highest score,
#     and computes intermediate activations.
#     Makes the visualization.
#############################################################################


def load_model(model_name='resnext101'):
    # Load the model
    model = torch.load('checkpoints/resnext101_32x8d_wsl_checkpoint.pth')
    model.fc = torch.nn.Linear(2048, n_classes)
    device = torch.device("cuda:{}".format(n_gpu - 1))
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)[::-1]), output_device=device)
    for param in model.parameters():
        param.requires_grad = True
    input_model_file = os.path.join(WORK_DIR, 'weights/model_{}_epoch{}_fold{}.bin'.format(WTSIZE, epoch, fold))
    model.load_state_dict(torch.load(input_model_file))
    model = model.to(device)
    model.eval()
    print(model.parameters())

    # Set config for the relevant model
    config = define_config()
    config = config[model_name]

    # Set up Grad-CAM
    grad_cam = GradCam(model=model.module,
                       pre_feature_block=config['pre_feature'],
                       feature_block=config['features'],  # features
                       target_layer_names=config['target'],
                       classifier_block=config['classifier'],  # classifier
                       use_cuda=torch.cuda.is_available())

    return model, grad_cam


################################
####  Load the input images  ###
################################


# Set parameters
class_idx = target_class_idx  # pick which subtype to visualise {0: EDH, 1: IPH, 2: IVH, 3: SAH, 4: SDH, 5: any}
label = label_cols[class_idx]  # label name (spelled fully)
target_index = class_idx  # set class_idx of target class. if set as None, returns the map for the model's predicted highest scoring class.
n_imgs = 5  # number of images to visualise (if imageidx_file not provided)

if imageidx_filename is not None:
    # If file containing image slice idx's is provided, use these indices.
    test = pd.read_csv(os.path.join(path_preds, imageidx_filename))
    img_idx = list(test['CT-idx_sliceidx'])
    test['Image'] = test['CT-idx_sliceidx']
    test = test.drop(labels=['CT-idx_sliceidx'], axis=1)
else:
    # No file containing image slice idx's provided, so:
    # Get idx of top n_imgs slices which the model predicted as containing [label] haemorrhage
    # by sorting y_score of [label] subclass by highest confidence

    y_score = pd.read_csv(os.path.join(path_preds, preds_name))
    y_score = y_score.sort_values(by=label, ascending=False)
    img_idx = list(y_score['CT-idx_sliceidx'][:n_imgs])
    test = y_score[:n_imgs].drop(labels=['Unnamed: 0'], axis=1)
    test['Image'] = test['CT-idx_sliceidx']
    test = test.drop(labels=['CT-idx_sliceidx'], axis=1)

path_img_raw = os.path.join(path_data, 'raw/')
path_img_brain = os.path.join(path_data, 'procbrain/')
path_img = os.path.join(path_data, 'proc/')

png = glob.glob(os.path.join(path_img, '*.jpg'))
png = [os.path.basename(png)[:-4] for png in png]
test_imgs = set(img_idx)
png = [p for p in png if p in test_imgs]
print('Number of images to visualise {}'.format(len(png)))
png = np.array(png)
test = test.set_index('Image').loc[png].reset_index()
print('Tst shape {} {}'.format(*test.shape))

mean_img = [0.22363983, 0.18190407, 0.2523437 ]
std_img = [0.32451536, 0.2956294,  0.31335256]
transform_test= Compose([
    HorizontalFlip(p=0.0),
    Transpose(p=0.0),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensorV2()
])

tstdataset = IntracranialDataset(test, path=path_img, transform=transform_test, labels=False)
num_workers = 0
tstloader = DataLoader(tstdataset, batch_size=1, shuffle=False, num_workers=num_workers)

for step, batch in enumerate(tstloader):
    model, grad_cam = load_model(model_name='resnext101')

    # Load each image one by one (batch_size=1)
    if step % 5 == 0:
        print('Tst step {} of {}'.format(step, len(tstloader)))
    imgnm = str(batch["img_name"][0])
    print(imgnm)
    input = batch["image"].to(device, dtype=torch.float)
    input = input.requires_grad_(True)  # note that this comes after transferring .to(device)
    print(type(input))
    print(input.size())
    print(input.grad)

    mask = grad_cam(input, target_index)

    # Create 'cam.jpg' image and overlay on raw image
    raw_img = cv2.imread(os.path.join(path_img_brain, imgnm + '.jpg'))
    if AUTOCROP:
        raw_img = autocrop(raw_img, threshold=0)
    raw_img = cv2.resize(raw_img, (SIZE, SIZE))
    cv2.imwrite(f'{imgnm}_raw.jpg', raw_img)
    show_cam_on_image(imgnm, label, raw_img, mask)

    # # Create 'gb.jpg' and 'cam_gb.jpg' image
    # gb_model = GuidedBackpropReLUModel(model=model.module, use_cuda=torch.cuda.is_available())
    # gb = gb_model(input, index=target_index)
    # gb = gb.transpose((1, 2, 0))    # convert to RGB
    # cam_mask = cv2.merge([mask, mask, mask])
    # cam_gb = deprocess_image(cam_mask*gb)
    # gb = deprocess_image(gb)
    # cv2.imwrite(f'{imgnm}_{label}_gb.jpg', gb)
    # cv2.imwrite(f'{imgnm}_{label}_cam_gb.jpg', cam_gb)
    #
    # # gb_on_raw = cv2.addWeighted(raw_img, 0.3, gb, 0.7, 0)
    # # cam_gb_on_raw = cv2.addWeighted(raw_img, 0.3, cam_gb, 0.7, 0)
    # # cv2.imwrite(f'{imgnm}_{label}_gb_on_raw.jpg', gb_on_raw)
    # # cv2.imwrite(f'{imgnm}_{label}_cam_gb_on_raw.jpg', cam_gb_on_raw)
    #
    # bitwise_and_images_adaptive(raw_img, gb, f'{imgnm}_{label}_gb_on_raw_adaptiveT.jpg')
    # bitwise_and_images_adaptive(raw_img, cam_gb, f'{imgnm}_{label}_cam_gb_on_raw_adaptiveT.jpg')
    #
    # bitwise_and_images(raw_img, gb, f'{imgnm}_{label}_gb_on_raw.jpg')
    # bitwise_and_images(raw_img, cam_gb, f'{imgnm}_{label}_cam_gb_on_raw.jpg')
