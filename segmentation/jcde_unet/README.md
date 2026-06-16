
<a id="readme-top"></a>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#project-structure">Project Structure</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#segmentation-pipeline-detection">Run MedSAM Detection for Segmentation</a></li>
        <li><a href="#segmentation-pipeline-medsam-segmentation">Run Segmentation Pipeline (MedSAM)</a></li>
        <li><a href="#run-mamba-mambaout-variants-for-segmentation">Run Mamba and MambaOut U-Net Variants for Segmentation</a></li>
        <li><a href="#visualisation-prerequisites">Run Visualisation</a></li>
      </ul>
    </li>
  </ol>
</details>



## About The Project

A Novel Patient-specific Heart Digital twin from CMR Images for Early Detection and Prognosis of Heart Abnormalities


## Project Structure

```
2023_FRGS_HeartDigitalTwin/
├── segmentation/
│   ├── v1-segmentation pipeline/                # Detection & MedSAM segmentation pipeline
│   │   ├── data/                                # Links to the raw and processed ACDC cardiac dataset
│   │   ├── weights/                             # YoloV11 model weights
│   │   ├── scripts/                             # Pre-process & segmentation script for Detection & MedSAM segmentation pipeline
│   │   └── requirements.txt    
│   │── v2-unet/                                 # U-net model (Mamba-Encoder U-Net)
│   │   ├── data/                                # Links to the ACDC, MMs & MMs-2 cardiac datasets
│   │   ├── models/                              # Model.py and weights
│   │   ├── utils/                               # pre-process script
│   │   └── requirements.txt         
│   └── jcde_unet/                               # Submission to JCDE Journal
│       ├── data/    
│       ├── datasets/                            # link-to-datasets.txt                       
│       ├── checkpoints/                         # llink-to-weights.txt          
│       ├── models/
│       │   ├── layers.py
|       │   ├── hybrid_mamba_unet.py             # hybrid Mamba U-Net         
|       │   ├── hybrid_mambaout_unet.py          # hybrid MambaOut U-Net
|       │   ├── pure_mamba_unet.py               # pure Mamba U-Net
|       │   ├── pure_mambaout_unet.py            # pure MambaOut U-Net
|       │   ├── pretrained_mamba_enc_unet.py     # pre-trained Mamba Encoder U-Net
|       │   └── pretrained_mambaout_enc_unet.py  # pre-trained MambaOut Encoder U-Net                         
│       ├── utils/    
│       ├── losses/                             
│       ├── metrics/     
│       ├── requirements.txt            
│       ├── config.py
│       ├── test.py
│       ├── train.py
│       └── pathology_analysis.py
│
└── visualisation/
    ├── node_modules/                            # Installed packages/dependencies
    ├── public/assets/                           # .obj Heart component models
    ├── src/
    │   ├── scripts/                 
    │   ├── AHA-plot.js                          # 2D LV plot
    │   ├── assign-color.js                      # Assign cardiac region color based on strain value
    │   ├── create-color-bar.js                  # Color bar scale for strain value
    │   ├── heart-model.js                       # 3D LV model
    │   └── main.js                              # Entry point
    ├── styles/
    │   └── style.css                         # CSS styles
    ├── index.html                            # HTML entry point
    ├── package.json                          # Project manifest
    └── package-lock.json                     # Project dependencies version (for reproduction)
```



<!-- GETTING STARTED -->
## Getting Started

For a local setup


### Segmentation Pipeline (Detection)
1. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

2. Run Inference
   ```sh
   yolo task=detect mode=predict \
      model=../weights/best.pt \
      conf=0.55 \
      source=../data/pre-processed/images \
      save=True \
      save_txt=True
   ```
### Segmentation Pipeline (MedSAM Segmentation)
1. Clone MedSAM repo
   ```sh
   git clone https://github.com/bowang-lab/MedSAM.git
   ```

2. Download the [MedSAM checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN) into folder i.e `checkpoints`

3. Run inference-pipeline.py
   ```sh
   python inference-pipeline.py \
       --detection_results_folder "../predict/labels" \
       --dataset_imgs_folder "../dataset" \
       --segmentation_results_folder "../segmentation" \
       --medsam_checkpoint_folder "../checkpoints/" \
       --medsam_weights "medsam_vit_b.pth"
   ```
---

### Run Mamba Mambaout Variants for Segmentation

### Installation

```sh
   git clone https://github.com/mmlchang/2023_FRGS_HeartDigitalTwin.git
   cd jcde_unet
   pip install -r requirements.txt
```
### Run Testing
#### 1. Download model weights in /checkpoints and cardiac datasets to /dataset (link provided in directory)

#### 2. Open config.py and set the weights and MODEL_ARCH variables to one of the six available variants

   ```sh
      # config.py
      WEIGHTS_DIR       = "../checkpoints/model"
      BEST_WEIGHTS_DIR  = "../checkpoints/model/best"

      MODEL_ARCH = "pretrainedmambaout"  
      # options:
      # pretrainedmamba, hybridmamba, hybridmambaout
      # puremamba, puremambaout
   ```

   #### 3. Run test.py

   ```sh
      python test.py --dataset acdc     # ACDC test set
      # OR python test.py --dataset mms      # M&Ms test set
      # OR python test.py --dataset mms2     # M&Ms-2 test set
   ```

   #### 4. Run Pathology Analysis
   Change the info_csv variable in pathology_analysis.py before running
   ```sh
      python pathology_analysis.py --dataset acdc # OR mms OR mms2
   ```

   ### Run Training
   ```sh
      python train.py    # default trained on ACDC test set
   ```

   ---

### Visualisation Prerequisites
- Node.js (v22.12.0 used) and npm (v9.8.1 used)

### Run Visualisation

1. Clone the repo
   ```sh
   git clone https://github.com/mmlchang/2023_FRGS_HeartDigitalTwin.git
   ```
2. Navigate via terminal
   ```sh
   cd 2023_FRGS_HeartDigitalTwin/visualisation
   ```
3. Install dependencies 
   ```sh
   npm install
   ```
4. Run visualisation
   ```sh
   npm start
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

