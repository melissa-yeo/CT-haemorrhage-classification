"""
Source: https://github.com/loveunk/pytorch-grad-cam/blob/7883e94c1b205d4f02a19b01bc9c04f4cfd6e72b/grad-cam.py#L41

Forked version of orig github repo by jacobgil, enabling support for resnet and other models.
To use other models, specify the architecture in the dict config.

"""

import argparse
import cv2
import json
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models


def define_config():
    config = {
        'vgg19': {
            'pre_feature': [],
            'features': 'features',
            'target': ['35'],
            'classifier': ['classifier']
        },
        'resnet50': {
            'pre_feature': ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3'],
            'features': 'layer4',
            'target': ['2'],
            'classifier': ['avgpool', 'fc']
        },
        'resnext101': {
            'pre_feature': ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3'],
            'features': 'layer4',
            'target': ['2'],
            'classifier': ['avgpool', 'fc']
        }
    }
    return config


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(imgnm, label, img, mask):
    '''

    :param imgnm: name of image for filename
    :param label: model's predicted label
    :param img: img (RGB, uint8, 0-255)
    :param mask: to be overlaid on img (0-1)
    :return:
    '''
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    # commented out the following line; it wasn't working with it
    # heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(f'{imgnm}_{label}_cam.jpg', np.uint8(255 * cam))


def bitwise_and_images(img1, img2, filename):
    # create a ROI that would fit onto img1
    rows, cols, _ = img2.shape
    roi = img1[0:rows, 0:cols]
    # create a mask of logo and its inverse mask
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('img2gray.jpg', img2gray)
    # plt.hist(img2gray)
    # plt.show()

    _, mask1 = cv2.threshold(img2gray, 155, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)  # gets the dark blue parts that got missed in mask1
    # cv2.imwrite('mask1.jpg', mask1)
    # cv2.imwrite('mask2.jpg', mask2)

    mask2 = cv2.bitwise_not(mask2)
    mask = mask1 + mask2
    # cv2.imwrite('mask.jpg', mask)

    # Erode away the finer elements. Erosion shrinks bright regions and enlarges dark regions.
    # Dilate to include some pixels surrounding the lung. Dilation enlarges bright regions and shrinks dark regions.
    eroded = cv2.erode(mask, np.ones((1, 1), np.uint8), iterations=3)
    dilation = cv2.dilate(eroded, np.ones((2, 2), np.uint8), iterations=1)
    mask = dilation

    # cv2.imwrite('mask_ed.jpg', mask)
    mask_inv = cv2.bitwise_not(mask)
    # cv2.imwrite('maskinv.jpg', mask_inv)

    # black-out the area of interest in ROI
    # this is the raw img with areas of interest blacked out, to be filled in
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # cv2.imwrite('img1_bg.jpg', img1_bg)

    # take only the region of interests from img2 heatmap, will be used to fill in blacked out holes in img1_bg
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # cv2.imwrite('img2_fg.jpg', img2_fg)

    # Put in the area of interest and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    cv2.imwrite(filename, img1)
    print('File created!')
    return


def bitwise_and_images_adaptive(img1, img2, filename):
    # create a ROI that would fit onto img1
    rows, cols, _ = img2.shape
    roi = img1[0:rows, 0:cols]
    # create a mask of logo and its inverse mask
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('img2gray.jpg', img2gray)
    img2gray = cv2.medianBlur(img2gray, 5)
    # cv2.imwrite('img2gray_blurred.jpg', img2gray)
    # plt.hist(img2gray)
    # plt.show()

    mask = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 3)
    # cv2.imwrite('mask.jpg', mask)

    # Erode away the finer elements. Erosion shrinks bright regions and enlarges dark regions.
    # Dilate to include some pixels surrounding the lung. Dilation enlarges bright regions and shrinks dark regions.
    eroded = cv2.erode(mask, np.ones((1, 1), np.uint8), iterations=3)
    dilation = cv2.dilate(eroded, np.ones((2, 2), np.uint8), iterations=1)
    mask = dilation

    # cv2.imwrite('mask_ed.jpg', mask)
    mask_inv = cv2.bitwise_not(mask)
    # cv2.imwrite('maskinv.jpg', mask_inv)

    # black-out the area of interest in ROI
    # this is the raw img with areas of interest blacked out, to be filled in
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # cv2.imwrite('img1_bg.jpg', img1_bg)

    # take only the region of interests from img2 heatmap, will be used to fill in blacked out holes in img1_bg
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # cv2.imwrite('img2_fg.jpg', img2_fg)

    # Put in the area of interest and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    cv2.imwrite(filename, img1)

    print('File created!')
    return


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, pre_features, features, target_layers):
        self.model = model
        self.pre_features = pre_features
        self.features = features
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        # forward pass through pre-feature block to get model output
        for pref in self.pre_features:
            x = getattr(self.model, pref)(x)

        # forward pass through feature block to get model output
        # also extract activations and register grads if at target layer in feature block
        submodel = getattr(self.model, self.features)
        for name, module in submodel._modules.items():
            # print(f'    inside FeatureExtractor: name: {name} ++++++++++ module: {module}')
            # print('++++++++++++++++++++++++++++++')
            x = module(x)
            if name in self.target_layers:
                # print(f'    inside FeatureExtractor: target_layer found. name: {name}')
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targeted layers.
    3. Gradients from intermediate targeted layers. """

    def __init__(self, model,
                 pre_feature_block=[],
                 feature_block='features',
                 target_layers='35',
                 classifier_block=['classifier']):
        self.model = model
        self.classifier_block = classifier_block
        # assume the model has a module named `feature`      ⬇⬇⬇⬇⬇⬇⬇⬇
        self.feature_extractor = FeatureExtractor(self.model,
                                                  pre_feature_block,
                                                  feature_block,
                                                  target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        # ⬇ target layer    ⬇ final feature layer's output after forward pass through pre-feature and feature blocks
        target_activations, output = self.feature_extractor(x)
        print('target_activations[0].size: {}'.format(target_activations[0].size()))  # for vgg'35 ([1, 512, 14, 14])
        print('output.size: {}'.format(output.size()))  # for vgg'36 ([1, 512, 7, 7])

        # forward pass through classifier layers to get final model output
        for i, classifier in enumerate(self.classifier_block):
            if i == len(self.classifier_block) - 1:
                # this is the final layer of the classifier block, for resnet it's the fc layer
                output = output.view(output.size(0), -1)
                print('output.view.size: {}'.format(output.size()))  # for vgg'36 ([1, 25088])

            output = getattr(self.model, classifier)(output)
            print('output.size: {}'.format(output.size()))  # for vgg'36 ([1, 1000])

        return target_activations, output


class GradCam:
    '''
    Forward pass the image though the model to obtain a raw score for each category.
    Compute the gradient of the score for class c, w.r.t feature map activations A^k of a conv layer.
        A^k is the final feature map (K is the total number of feature maps).
    Average all these gradient scores.
    Compute GradCAM by applying RELU (only accepts positive values).
    '''
    def __init__(self, model, pre_feature_block, feature_block, target_layer_names, classifier_block, use_cuda):
        self.model = model
        self.model.eval()
        self.pre_feature_block = pre_feature_block
        self.feature_block = feature_block
        self.classifier_block = classifier_block
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model,
                                      pre_feature_block,
                                      feature_block,
                                      target_layer_names,
                                      classifier_block)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        # get target layer activations and model output; grads from relevant layers are now registered
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        # set the target class index if None was provided
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        # create one_hot which zeroes outputs from all other class indices except for target class
        print('output.size: {}'.format(output.size()))  # (1, 6)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # zero all gradients in the feature and classifier blocks, and backprop to compute gradients
        # make sure graph is retained so memory of computation graph is not discarded after computation (this happens by default after .backward)
        # print('output: {}'.format(output))
        # print('one_hot: {}'.format(one_hot))
        getattr(self.model, self.feature_block).zero_grad()
        for classifier in self.classifier_block:
            getattr(self.model, classifier).zero_grad()
        one_hot.backward(retain_graph=True)

        # get the gradients for the given class w.r.t the final feature map
        gradients = self.extractor.get_gradients()
        # print('len(gradients): {}'.format(len(gradients)))
        print('gradients[0].size(): {}'.format(gradients[0].size()))    # (1, 2048, 15, 15)
        print('gradients[-1].size(): {}'.format(gradients[-1].size()))
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        print('grads_val.amax: {}'.format(np.amax(grads_val)))

        # get the target activations (final feature map?)
        target = features[-1]
        print('target.size(): {}'.format(target.size()))   # (1, 2048, 15, 15)
        target = target.cpu().data.numpy()[0, :]
        print('target.shape: {}'.format(target.shape))   # (2048, 15, 15)
        print('target.amax: {}'.format(np.amax(target)))

        # average gradients spatially (the avg of each filter's weights)
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        print('weights.shape: {}'.format(weights.shape))    # (2048,)
        print('weights.amax: {}'.format(np.amax(weights)))

        # final GradCAM has same shape as final conv feature map
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        print('cam.shape: {}'.format(cam.shape))  # (15, 15)

        # for each of the 2048 weights, multiply by the feature map
        # compute the successive matrix products of: the weight matrices, and the grad w.r.t activation functions
        # until the final conv layer that the gradients are being propagated to.
        # cam is the summation of these calculations
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]  # (15,15) += (1) * (15,15)
        # print('cam: {}'.format(cam))
        print('cam.shape: {}'.format(cam.shape))    # (15, 15)
        cam = np.maximum(cam, 0)  # remove negative numbers
        print('cam.amax: {}'.format(np.amax(cam)))
        cam = cv2.resize(cam, input.shape[2:])  # resize to same size as input img
        print('cam.shape: {}'.format(cam.shape))    # (480, 480)
        # with np.printoptions(threshold=np.inf):
        #     print(cam)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        # cam - (cam - cam.min()) / (cam.max() - cam.min())   # alternative standardisation process
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

        # # replace ReLU with GuidedBackpropReLU
        # for idx, module in self.model.features._modules.items():
        #     if module.__class__.__name__ == 'ReLU':
        #         self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        # forward pass to get model output
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        # get target class index
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        # create one_hot which zeroes outputs from all other class indices except for target class
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def deprocess_image(img):
    """
    utility function to convert a float array into a valid uint8 image.
    see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65
    """
    # normalize tensor: center on 0., ensure std is 0.1
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    # clip to [0, 1]
    img = img + 0.5
    img = np.clip(img, 0, 1)
    # convert to (0-255)
    return np.uint8(img * 255)
