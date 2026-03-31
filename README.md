
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
        <li><a href="#segmentation-pipeline">Prerequisites</a></li>
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
│   ├── v1-segmentation pipeline/             # Detection & MedSAM segmentation pipeline
│   │   ├── data/                             # Links to the raw and processed ACDC cardiac dataset
│   │   ├── weights/                          # YoloV11 model weights
│   │   ├── scripts/                          # Pre-process & segmentation script for Detection & MedSAM segmentation pipeline
│   │   └── requirements.txt    
│   └── v2-unet/                              # U-net model (Mamba-Encoder U-Net)
│       ├── data/                             # Links to the ACDC, MMs & MMs-2 cardiac datasets
│       ├── models/                           # Model.py and weights
│       ├── utils/                            # pre-process script
│       └── requirements.txt                
└── visualisation/
    ├── node_modules/                         # Installed packages/dependencies
    ├── public/assets/                        # .obj Heart component models
    ├── src/
    │   ├── scripts/                 
    │   ├── AHA-plot.js                       # 2D LV plot
    │   ├── assign-color.js                   # Assign cardiac region color based on strain value
    │   ├── create-color-bar.js               # Color bar scale for strain value
    │   ├── heart-model.js                    # 3D LV model
    │   └── main.js                           # Entry point
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
```
  pip install -r requirements.txt
```

2. Run Inference
```
  yolo task=detect mode=predict \
      model=./weights/best.pt \
      conf=0.55 \
      source=./data/pre-processed/images \
      save=True \
      save_txt=True
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

