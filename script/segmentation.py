# -*- coding: utf-8 -*-

import os
join = os.path.join
import sys

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F

import nibabel as nib

if len(sys.argv) > 1:
    PATH_TO_MEDSAM = sys.argv[1]  # 1st arg after script name
else:
    sys.exit(1)

os.chdir(PATH_TO_MEDSAM)

def create_segmentation_results_folder(given_filepath):
  seg_results_folder = given_filepath
  rv_fold = os.path.join(seg_results_folder, 'rv')
  myo_fold = os.path.join(seg_results_folder, 'myo')
  lv_fold = os.path.join(seg_results_folder, 'lv')

  if not os.path.exists(seg_results_folder):
    os.makedirs(seg_results_folder, exist_ok=True)
    os.makedirs(rv_fold, exist_ok=True)
    os.makedirs(myo_fold, exist_ok=True)
    os.makedirs(lv_fold, exist_ok=True)
    print('Folder created')
  else:
    print('Folder already created')

def get_yolo_coordinates(label_filename):
    # Class, xcen, ycen, wnorm, hnorm -> yolo compatible
    # 0 == rv, 1 == myo, 2 == rv (yolov5 class labels must start from 0!

    bbox_list = {'class':[], 'x_cen':[], 'y_cen':[], 'w_norm':[], 'h_norm':[]}

    with open(label_filename, 'r') as file:
        for line in file:
            l_list = line.split()
            bbox_list['class'].append(int(l_list[0]))
            bbox_list['x_cen'].append(float(l_list[1]))
            bbox_list['y_cen'].append(float(l_list[2]))
            bbox_list['w_norm'].append(float(l_list[3]))
            bbox_list['h_norm'].append(float(l_list[4]))

    return bbox_list

def yolov5_bbox_to_absolute(img_arr, lab_class, x, y, w, h): # given image in array
    #img = mpimg.imread(img)
    img_height, img_width = img_arr.shape[:2]

    x_center_abs = x * img_width
    y_center_abs = y * img_height
    width_abs = w * img_width
    height_abs = h * img_height

    x_min = x_center_abs - width_abs / 2
    y_min = y_center_abs - height_abs / 2
    x_max = x_center_abs + width_abs / 2
    y_max = y_center_abs + height_abs / 2

    return x_min, y_min, x_max, y_max

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

# LOAD MEDSAM MODEL
def define_medsam_model(mod_path, weights="medsam_vit_b.pth"):
  MedSAM_CKPT_PATH = os.path.join(mod_path, weights)
  device = "cuda:0"
  medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
  medsam_model = medsam_model.to(device)
  # medsam_model.eval()
  print('Model Loaded')
  return device, medsam_model

# SEGMENTATION AND SAVE OUTPUT
def segment_and_save_mask(device, medsam_model, detection_results_folder, dataset_imgs_folder, label_txt_file, segmentation_results_folder):

  yolo_coord = get_yolo_coordinates(os.path.join(detection_results_folder, label_txt_file))

  nifti_file = label_txt_file.replace('.txt', '')

  if "ES" in label_txt_file:
    nifti_file = (nifti_file.replace('ES', ''))

  if "ED" in label_txt_file:
    nifti_file = (nifti_file.replace('ED', ''))

  [patientid, sliceidx, timeframeidx] = nifti_file.split('_')
  nifti_file = f"{patientid}_4d.nii.gz"
  nifti_path = os.path.join(dataset_imgs_folder, nifti_file)

  nii_img = nib.load(nifti_path)
  nii_data = nii_img.get_fdata()

  img_height, img_width, number_of_slices, number_of_frames = nii_data.shape
  orgsize_img_from_nifti = nii_data[:, :, int(sliceidx), int(timeframeidx)]
  img_from_nifti = cv2.resize(orgsize_img_from_nifti, (640, 640), interpolation=cv2.INTER_LANCZOS4)
  #plt.imshow(img_from_nifti, cmap='gray')

  # NOTE: it is better to read the slice data straight from the nifti file for segmentation rather than image format
  # image label format is in patientid_(slice)_(timeframe).txt
  # can split the file by '_' and substitute the indexes into nii[:, :, slice, timeframe]

  yolo_coord = get_yolo_coordinates(os.path.join(detection_results_folder, label_txt_file))
  df = pd.DataFrame(yolo_coord)

  seg = {"lvc": np.array([]), "rv": np.array([]), "myo": np.array([])}

  for n in range(0, len(df)):
    lab_class = df['class'][n]
    x_cen = df['x_cen'][n]
    y_cen = df['y_cen'][n]
    w_norm = df['w_norm'][n]
    h_norm = df['h_norm'][n]

    target_coord = yolov5_bbox_to_absolute(img_from_nifti, lab_class, x_cen, y_cen, w_norm, h_norm)

    img_3c = cv2.normalize(img_from_nifti, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    if lab_class == 1: # if it is lv myo - slightly different processing
      if len(img_3c.shape) == 3:
        img_3c = cv2.cvtColor(img_3c, cv2.COLOR_BGR2GRAY)
      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
      img_3c = 255 - clahe.apply(img_3c) # inverted with increased contrast

    if len(img_3c.shape) == 2:
      img_3c = np.repeat(img_3c[:, :, None], 3, axis=-1)

    H, W, _ = img_3c.shape

    #%% image preprocessing and model inference
    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)

    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)

    # convert the shape to (3, H, W)
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    box_np = np.array([target_coord]) #[xmin, ymin, xmax, ymax]

    # transfer box_np t0 1024x1024 scale
    box_1024 = box_np / np.array([W, H, W, H]) * 1024

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor) # (1, 256, 64, 64)

    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
    # resize segmentation mask
    medsam_seg = cv2.resize(medsam_seg, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

    if lab_class == 0:
      seg['rv'] = medsam_seg
    elif lab_class == 1:
      seg['myo'] = medsam_seg
    else: # lab_class == 2
      seg['lvc'] = medsam_seg

  # invert and obtain lvc myo mask
  if seg['lvc'].size != 0:
    arr_lvc_invert = 1 - seg['lvc']
    seg['myo'] = seg['myo'] * arr_lvc_invert

  # save segmentation masks to .csv format
  mask_csv = f"{patientid}_{sliceidx}_{timeframeidx}.csv"
  if seg['lvc'].size != 0:
    df = pd.DataFrame(seg['lvc'])
    df.to_csv(os.path.join(segmentation_results_folder, 'lv', mask_csv), index=False, header=False)

  if seg['rv'].size != 0:
    df = pd.DataFrame(seg['rv'])
    df.to_csv(os.path.join(segmentation_results_folder, 'rv', mask_csv), index=False, header=False)

  if seg['myo'].size != 0:
    df = pd.DataFrame(seg['myo'])
    df.to_csv(os.path.join(segmentation_results_folder, 'myo', mask_csv), index=False, header=False)

  return orgsize_img_from_nifti, seg #return org nifti slice array & segmentation masks