#!/usr/bin/env python
# coding: utf-8

# In[115]:


import os
import nibabel as nib
import cv2


def normalize_and_resize(input_nifti, id, k, save_folder): # pre-process data. Normalize and Resize to (k x k x 3)

    output = {"img_name": [], "org_img_width": [], "org_img_height": []}

    nii_img = nib.load(input_nifti)
    nii_data = nii_img.get_fdata()
    
    # load image (4D) [X, Y, Z_slice, frame]
    img_width = nii_data.shape[0]
    img_height = nii_data.shape[1]
    number_of_slices = nii_data.shape[2]
    number_of_frames = nii_data.shape[3]

    for frame in range(number_of_frames):

        #fig, axs = plt.subplots(1, number_of_slices, figsize=(25, 5))
        #for slice, ax in enumerate(axs.flat): # For every slice print the image otherwise show empty space.

            if slice < number_of_slices:

                img_normalize = cv2.normalize(nii_data[:, :, slice, frame], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) #float64
                img_normalize = img_normalize.astype('uint8') #convert to uint8
                img = cv2.resize(img_normalize, (k, k), interpolation=cv2.INTER_LANCZOS4)

                if len(img.shape) == 2: # convert to rgb 3 channels since yolo works with rgb
                    img = cv2.merge([img, img, img]) 

                img_name = f"{id}_{frame}_{slice}.jpg"

                img_output['img_name'].append(img_name)
                img_output['org_img_width'].append(img_width)
                img_output['org_img_height'].append(img_height)

                #cv2.imwrite(os.path.join(save_folder, img_name), img) #resized

    #             ax.imshow(png, interpolation=None)
    #             ax.set_title(f"slice {slice} / frame {frame}")
    #             ax.axis('off')
    #        else:
    #             ax.axis('off')

    #     plt.pause(0.05)
    #     plt.close(fig)
    return img_output



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

def yolocoordinates_to_rect(class_lab, x_cen, y_cen, w_norm, h_norm, img_w, img_h):
    # Class, xcen, ycen, wnorm, hnorm -> yolo compatible
    # 0 == rv, 1 == myo, 2 == rv (yolov5 class labels must start from 0!)
    x_cen = float(x_cen)
    y_cen = float(y_cen)
    w_norm = float(w_norm)
    h_norm = float(h_norm)
    img_w = int(img_w)
    img_h = int(img_w) # note python converts int into float for multiplication later

    rect_x = (x_cen - (w_norm/2))*img_w
    rect_y = (y_cen - (h_norm/2))*img_h

    rect_w = w_norm * img_w
    rect_h = h_norm * img_h

    bbox = {'class':class_lab, 'rect_x':rect_x, 'rect_y':rect_y, 'rect_w':rect_w, 'rect_h':rect_h}

    return bbox


def yolocoordinates_to_torchcoord(class_lab, x_cen, y_cen, w_norm, h_norm, img_w, img_h):
    # Class, xcen, ycen, wnorm, hnorm -> yolo compatible
    # 0 == rv, 1 == myo, 2 == rv (yolov5 class labels must start from 0!)
    x_cen = float(x_cen)
    y_cen = float(y_cen)
    w_norm = float(w_norm)
    h_norm = float(h_norm)
    img_w = int(img_w)
    img_h = int(img_w) # note python converts int into float for multiplication later

    xmin = (x_cen - (w_norm/2))*img_w
    ymin = (y_cen - (h_norm/2))*img_h

    xmax = (x_cen + (w_norm/2))*img_w
    ymax = (y_cen + (h_norm/2))*img_h

    bbox = {'class':class_lab, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax}

    return bbox



def mask_non_cardiac(img, label_txt, save_folder, id):

    with open(label_txt, 'r') as file:
        labels = file.readline()
    labels = labels.split(' ')

    if labels != ['']:
        img = cv2.imread(img)
        img = np.array(img)

        rect = yolocoordinates_to_rect(int(labels[0]), float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4]), img.shape[1], img.shape[0])

        mask = (img.copy()).astype(np.uint8)
        x = int(rect['rect_x'])
        y = int(rect['rect_y'])
        w = int(rect['rect_w'])
        h = int(rect['rect_h'])

        x1 = x 
        y1 = y
        x2 = x+w
        y2 = y+h

        # Set all pixels outside the bounding box to 1
        mask[:y1, :] = 255.  # top  
        mask[y2:, :] = 255.  # bottom 
        mask[:, :x1] = 255.  # left 
        mask[:, x2:] = 255.  # right 
        
        img_name = f"masked_{id}.jpg"

        cv2.imwrite(os.path.join(save_folder, img_name), mask) #resized
        #print(f"Saved to {os.path.join(save_folder, img_name)}")
        #plt.imshow(mask)
        #plt.show()
        
    return img_name

