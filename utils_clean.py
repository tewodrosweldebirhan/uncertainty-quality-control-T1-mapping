'''Utils contains some useful functions for training and testing.
It contains some functions used for image level uncertainty estimation such as Dice and HD within samples
'''

import os.path

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss, accuracy_score
from scipy.ndimage import binary_dilation, generate_binary_structure, binary_erosion
# from datasets.dataset_synapse import get_spatial_class
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_pickle, load_pickle


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class DiceCoeff(nn.Module):
    def __init__(self, n_classes):
        super(DiceCoeff, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return class_wise_dice


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def calculate_metric_perslice(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def calculate_metric_perslice_fullhd(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        myhd = metric.binary.hd(pred, gt)
        return dice, myhd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def dice_coef_metric(pred, target):
    """This definition generalize to real valued pred and target vector.
        This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
    """
    smooth = 1.
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    dice_score = ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

    return dice_score.numpy()


def load_itk(filename):
    """
    Convert the image to a numpy array first
     and then shuffle the dimensions to get axis in the order z,y,x
    """
    itk_img = sitk.ReadImage(filename)
    np_img = sitk.GetArrayFromImage(itk_img)
    return itk_img, np_img


def reconstruct_nii(cropped_image, image_size, center_coordinate, patch_size=128):
    reconstructed_image = np.zeros(image_size, dtype=np.uint8)
    height = patch_size // 2
    ccenter_x, ccenter_y = center_coordinate[0], center_coordinate[1]
    reconstructed_image[ccenter_x - height: ccenter_x + height, ccenter_y - height: ccenter_y + height] = cropped_image
    reconstructed_image = np.expand_dims(reconstructed_image, axis=0)

    return reconstructed_image

def save_prediciton_nii(pred_mean_final, label, label_slice,case_name_new):
    # save nifti images of the predicted segmentation
    # resize the seg to the original size
    pred_mean_final_sq = pred_mean_final[0, 0, :, :]
    pred_mean_final_resized = zoom(pred_mean_final_sq, (
        label.shape[0] / label_slice.shape[0], label.shape[1] / label_slice.shape[1]), order=0)
    print(pred_mean_final_resized.shape)
    # read the original GT
    orig_gt_path = '../Inhouse_dataset/RV_DATASETS/01_centered/'
    test_result_path = os.path.join('../Inhouse_dataset/PostContrast_Test_Results_All/', case_name_new)
    maybe_mkdir_p(test_result_path)
    specific_path_img = '*/*/' + case_name_new + '.nii.gz'
    specific_path = '*/*_CONTOURS/' + case_name_new + '.nii.gz'
    import glob
    gt_path_nii = list(sorted(glob.glob(orig_gt_path + specific_path)))
    img_path_nii = gt_path_nii[0].replace("_CONTOURS", "")
    print(gt_path_nii)
    print(img_path_nii)

    itk_img_gt, img_gt = load_itk(gt_path_nii[0])
    itk_img_img, img_img = load_itk(img_path_nii)

    center_x = img_gt.shape[1] // 2
    center_y = img_gt.shape[2] // 2
    orig_img_size = (img_gt.shape[1], img_gt.shape[2])
    orig_img_center = (center_x, center_y)
    # further resizing to the original gt size
    reconst_prediction = reconstruct_nii(pred_mean_final_resized, orig_img_size, orig_img_center)
    print(reconst_prediction.shape)
    print(img_gt.shape)

    # read the meta info of the original GT
    itk_pred = sitk.GetImageFromArray(reconst_prediction, isVector=False)
    itk_pred.CopyInformation(itk_img_gt)

    pred_fname = os.path.join(test_result_path, case_name_new + '_pred.nii.gz')
    gt_fname = os.path.join(test_result_path, case_name_new + '_gt.nii.gz')
    img_fname = os.path.join(test_result_path, case_name_new + '_img.nii.gz')
    print(pred_fname)
    print(gt_fname)
    print(img_fname)
    sitk.WriteImage(itk_pred, pred_fname, True)
    sitk.WriteImage(itk_img_gt, gt_fname, True)
    sitk.WriteImage(itk_img_img, img_fname, True)

    #

def dice_hd_within_samples(pred_samples, classes, class_weights=[0.0, 1.0, 1.0, 1.0]):
    # Image level uncertainty map: Dice_withinsamples = Mean of the Dice coefficients between mean segmentation mask and each sample
    num_samples = len(pred_samples)
    pred_mean = np.mean(pred_samples, axis=0, keepdims=True)  # samples dims (n=5, 3, 96, 96)
    #
    pred_mean = torch.from_numpy(pred_mean)
    pred_mean_argmax = pred_mean.max(1, keepdim=True)[1]
    pred_mean_argmax = pred_mean_argmax.numpy()
    Total_dicesample_one_subject = 0
    Total_hdsample_one_subject = 0
    for samp_no in range(num_samples):

        sample_torch = torch.from_numpy(np.expand_dims(pred_samples[samp_no], axis=0))
        sample_torch_argmax = sample_torch.max(1, keepdim=True)[1]
        sample_torch_argmax = sample_torch_argmax.numpy()

        sum_hd_one_sample = 0
        sum_dice_one_sample = 0
        for cn in range(1, classes): # exclude the background
            sum_dice_one_sample += (
                    dice_coef_metric(sample_torch[:, cn, :, :], pred_mean[:, cn, :, :]) * class_weights[cn])

            sum_hd_one_sample += (
                    calculate_metric_perslice_fullhd(sample_torch_argmax == cn, pred_mean_argmax == cn)[1] *
                    class_weights[cn])  # ? hd

        Total_dicesample_one_subject += (sum_dice_one_sample / (classes - 1))  # without the background
        Total_hdsample_one_subject += (sum_hd_one_sample / (classes - 1))  # without the background

    Dice_withinsamples = (Total_dicesample_one_subject / num_samples)
    HD_withinsamples = (Total_hdsample_one_subject / num_samples)

    return Dice_withinsamples, HD_withinsamples

def dice_hd_within_samples_perstructure(pred_samples, classes, class_weights=[0.0, 1.0, 1.0, 1.0]):
    # Image level uncertainty map: Dice_withinsamples = Mean of the Dice coefficients between mean segmentation mask and each sample
    num_samples = len(pred_samples)
    pred_mean = np.mean(pred_samples, axis=0, keepdims=True)  # samples dims (n=5, 3, 96, 96)
    #
    pred_mean = torch.from_numpy(pred_mean)
    pred_mean_argmax = pred_mean.max(1, keepdim=True)[1]
    pred_mean_argmax = pred_mean_argmax.numpy()
    Total_dicesample_one_subject = np.zeros(shape=(num_samples,classes-1), dtype=np.float16)
    Total_hdsample_one_subject = np.zeros(shape=(num_samples,classes-1), dtype=np.float16)
    for samp_no in range(num_samples):

        sample_torch = torch.from_numpy(np.expand_dims(pred_samples[samp_no], axis=0))
        sample_torch_argmax = sample_torch.max(1, keepdim=True)[1]
        sample_torch_argmax = sample_torch_argmax.numpy()

        classwise_hd_one_sample = []
        classwise_dice_one_sample = []
        for cn in range(1, classes): # exclude the background
            Total_dicesample_one_subject[samp_no, cn-1] = (
                    dice_coef_metric(sample_torch[:, cn, :, :], pred_mean[:, cn, :, :]) * class_weights[cn])

            Total_hdsample_one_subject[samp_no, cn-1] = (
                    calculate_metric_perslice_fullhd(sample_torch_argmax == cn, pred_mean_argmax == cn)[1] *
                    class_weights[cn])  # ? hd


    Dice_withinsamples = np.mean(Total_dicesample_one_subject,axis=0) #compute mean along the no of samples axis
    HD_withinsamples = np.mean(Total_hdsample_one_subject, axis=0) #compute mean along the no of samples axis

    return Dice_withinsamples, HD_withinsamples