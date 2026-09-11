# SmartPPE — Safwan

**Industrial Applications of Computer Vision — A.A. 2025–26**  
**Università degli Studi di Padova**

**Author:** Syed Muhammad Safwan

**GitHub repository:** https://github.com/smsafwan/SmartPPE_Safwan

---

## 1. Project Overview

SmartPPE is a computer vision system for monitoring Personal Protective Equipment (PPE) compliance in an industrial environment.

The system takes an RGB video stream and detects five PPE-related conditions:

- `Bare_Head`
- `No_Safety_vest`
- `Non_Compliant_Hat`
- `Safety_vest`
- `helmet`

The final detector is an **RT-DETR-L** model trained for the PPE detection task. The final model was trained at **640 × 640** input resolution and the tested production configuration uses the trained PyTorch weights (`best.pt`).

When a violation is detected, the system can show it on the video interface, trigger a local alarm and prepare a Telegram notification with a snapshot of the detected frame.

The production pipeline is asynchronous. Video capture, inference and display are handled separately so that the display remains responsive even though the detector runs slower than the camera/display loop.

---

## 2. Problem Definition

The aim of the project is to detect PPE compliance violations from a live RGB camera stream.

| Class | Meaning | Type |
|---|---|---|
| `Bare_Head` | Worker without appropriate head protection | Violation |
| `No_Safety_vest` | Worker without a safety vest | Violation |
| `Non_Compliant_Hat` | Non-compliant headwear | Violation |
| `Safety_vest` | Safety vest detected | Compliant |
| `helmet` | Safety helmet detected | Compliant |

The system is intended for visible-light industrial environments. It is not designed for complete darkness, infrared-only operation, cross-camera worker re-identification, or tracking workers between separate non-overlapping camera views.

---

## 3. Project Targets

The original project proposal defined the following main performance targets:

| Metric | Target |
|---|---:|
| Inference speed | ≥ 25 FPS |
| mAP@0.50 | ≥ 0.80 |
| Precision | ≥ 0.75 |
| Recall | Maximize |

Recall was considered especially important because a missed PPE violation is more serious for this application than an additional false alarm.

The final model exceeded the mAP@0.50 and precision targets, but the tested model inference speed did **not** reach the original 25 FPS target. The measured model inference speed was approximately **6.5 FPS**.

---

## 4. System Input and Output

### Input

The main input is a live RGB video stream from a webcam or camera.

- Input type: RGB video frames
- Model input resolution: **640 × 640**
- Object detection model: RT-DETR-L
- Production model: trained PyTorch `best.pt`

### Output

The system produces:

1. Bounding boxes around detected PPE-related objects.
2. Class labels.
3. Detection confidence values.
4. Local audio warnings for PPE violations.
5. A violation snapshot for remote notification.
6. Telegram notification functionality when configured.

A 5-second debounce is used for Telegram notifications so that the same ongoing violation does not continuously generate messages.

---

## 5. Repository Structure

```text
SmartPPE_Safwan/
│
├── assets/
│   ├──── original-images/
│   ├──── sample_compliant.jpg
│   ├──── sample_violation-1.jpg
│   ├──── sample_violation-2.jpg
│   ├──── sample_violation-3.jpg
│   └──── alarm.wav
│
├── datasets/
│   
│
├── docs/
│   └── SmartPPE_Safwan.pdf
│
├── src/
│   │── evaluate_metrics.py
│   ├── production_pipeline.py
│   ├── retrain_model.py
│   ├── run_ppe_detection.py
│   ├── test_env.cpp
│   ├── train_model.py
│   └── train_new.py
│
├── weights/
│   └── best.pt
│
├── .gitignore
└── README.md
```

### Repository components

| Component | Purpose |
|---|---|
| `assets/` | Sample results and local alarm audio |
| `docs/` | Final project presentation |
| `src/` | Codes of Training, Reetraining and Production |
| `weights/` | Final trained PyTorch model |
| `.gitignore` | Prevents unwanted files and local artifacts from being committed |
| `README.md` | Project documentation and reproducibility information |

 Datasets are not stored directly in the GitHub repository. Dataset access is provided through external links.

---

## 6. Dataset

Three main datasets were used during the project:

1. Training dataset
2. Retraining dataset
3. Independent evaluation dataset

The training and retraining data were assembled from images obtained from different datasets available through Roboflow. The independent evaluation data were obtained from several YouTube videos and were kept separate from the training data.

### Dataset links

#### Training dataset

Google Drive:

https://drive.google.com/file/d/1e5Nt4mfDA74fdtsChdgW0-7QM7NlhAOz/view?usp=drive_link


#### Retraining / fine-retraining dataset

Google Drive:

https://drive.google.com/file/d/1O0wosXzzpDyOKUwNkCrDU9Upw6GiNv67/view?usp=sharing

Roboflow source datasets used to create datasets for Training and Retraining:

```text
https://universe.roboflow.com/john-lemuel-tapel-ptu3s/safety-vest-detection-2fwgw
https://universe.roboflow.com/aoqawkis/-cap
https://universe.roboflow.com/elte-dnd-e5jud/traffic-cone-k9gdu
https://universe.roboflow.com/ashish-chouhan/bright-cubes/
https://universe.roboflow.com/pxl/objects-m9q64
https://universe.roboflow.com/new-workspace-ae1ks/safety-helmet-mp2do
https://universe.roboflow.com/pavithrap/cap-jby6j
https://universe.roboflow.com/dpuw/yellow-ptqe8

```

#### Evaluation dataset

Google Drive:

https://drive.google.com/file/d/1LpDP9M3984Qyh3PxFZ-m3ejX5ITx9mGy/view?usp=drive_link

The evaluation dataset was created from scratch and independent from the training and retraining data, using frames obtained from the following YouTube sources:

1. https://www.youtube.com/watch?v=5TChgJ3781M
2. https://www.youtube.com/watch?v=Pm_p6urP7Wg
3. https://www.youtube.com/watch?v=Dhxf5mm7g1g
4. https://www.youtube.com/watch?v=3eCKfBkSUMU
5. https://www.youtube.com/watch?v=4Vpee2sEUDs
6. https://www.youtube.com/watch?v=lfoTLeFooR4
7. https://www.youtube.com/watch?v=InA7r_JUndk
8. https://www.youtube.com/watch?v=lifunadBZ3U
9. https://www.youtube.com/watch?v=5V6R_swdLGI
10. https://www.youtube.com/watch?v=A0cXKa4Y3do
11. https://www.youtube.com/watch?v=_dDuKCQP9Xc
12. https://www.youtube.com/watch?v=Bjuwpo9d3cQ

The evaluation set was not taken from the training datasets.

---

## 7. Dataset Statistics

The numbers below are **annotated object instances**, not numbers of images.

### Training dataset

Total annotated instances: **2,085**

| Class | Instances |
|---|---:|
| `Bare_Head` | 353 |
| `No_Safety_vest` | 425 |
| `Non_Compliant_Hat` | 377 |
| `Safety_vest` | 386 |
| `helmet` | 544 |
| **Total** | **2,085** |

### Retraining dataset

Total annotated instances: **1,488**

| Class | Instances |
|---|---:|
| `Bare_Head` | 98 |
| `No_Safety_vest` | 495 |
| `Non_Compliant_Hat` | 60 |
| `Safety_vest` | 224 |
| `helmet` | 611 |
| **Total** | **1,488** |

### Evaluation dataset

Total annotated instances: **202**

| Class | Instances |
|---|---:|
| `Bare_Head` | 15 |
| `No_Safety_vest` | 65 |
| `Non_Compliant_Hat` | 17 |
| `Safety_vest` | 37 |
| `helmet` | 68 |
| **Total** | **202** |


---

## 8. Ground Truth and Annotation

The datasets was manually annotated with bounding boxes for the five target classes.

The annotation classes were:

```text
Bare_Head
No_Safety_vest
Non_Compliant_Hat
Safety_vest
helmet
```

After the initial annotation process, **every annotated instance was personally inspected by the project author**. Incorrect bounding boxes, class assignments and missing or incorrect annotations were corrected before the datasets were used for training or evaluation.

This manual verification was important because the evaluation dataset is used as the ground truth reference when comparing the different model runs.

---

## 9. Model and Training

The project uses the **RT-DETR-L** object detection architecture through Ultralytics.

The project went through four main experimental stages:

1. Baseline
2. Retraining 1
3. Retraining 2
4. Final model

The same RT-DETR-L model family was used throughout the experiments.

The final model was initialized from the best weights obtained during Retraining 2 and then retrained using the final dataset/configuration.

---

## 10. Training Configurations

### 10.1 Baseline

| Parameter | Value |
|---|---:|
| Model | RT-DETR-L |
| Initial weights | `rtdetr-l.pt` |
| Dataset | `datasets/ppe_dataset/data.yaml` |
| Epochs | 30 |
| Patience | 100 |
| Batch size | 2 |
| Image size | 640 × 640 |
| Cache | True |
| Workers | 2 |
| Close mosaic | 10 |
| Initial learning rate (`lr0`) | 0.01 |
| Final learning rate factor (`lrf`) | 0.01 |
| Warmup epochs | 3 |
| IoU | 0.7 |
| Box loss weight | 7.5 |
| Classification loss weight | 0.5 |
| HSV-H | 0.015 |
| HSV-S | 0.7 |
| HSV-V | 0.4 |
| Scale | 0.5 |
| Mosaic | 1.0 |
| Horizontal flip | 0.5 |
| Auto augmentation | RandAugment |
| Erasing | 0.4 |
| AMP | True |
| Optimizer | Auto |
| Pretrained | True |
| Seed | 0 |
| Deterministic | True |

### 10.2 Retraining 1

| Parameter | Value |
|---|---:|
| Model | RT-DETR-L |
| Initial weights | `rtdetr-l.pt` |
| Dataset | `datasets/ppe_dataset/data.yaml` |
| Epochs | 40 |
| Patience | 100 |
| Batch size | 2 |
| Image size | 800 × 800 |
| Cache | True |
| Workers | 2 |
| Close mosaic | 10 |
| Initial learning rate (`lr0`) | 0.01 |
| Final learning rate factor (`lrf`) | 0.01 |
| Warmup epochs | 3 |
| IoU | 0.7 |
| Box loss weight | 7.5 |
| Classification loss weight | 0.5 |
| HSV-S | 0.7 |
| HSV-V | 0.4 |
| Scale | 0.5 |
| Mosaic | 1.0 |
| Horizontal flip | 0.5 |
| Auto augmentation | RandAugment |
| Erasing | 0.4 |

### 10.3 Retraining 2

| Parameter | Value |
|---|---:|
| Model | RT-DETR-L |
| Dataset | `datasets/ppe_balanced_v2/data.yaml` |
| Epochs | 80 |
| Patience | 20 |
| Batch size | 4 |
| Image size | 600 × 600 |
| Cache | False |
| Device | `0` |
| Workers | 0 |
| Close mosaic | 15 |
| Initial learning rate (`lr0`) | 0.01 |
| Final learning rate factor (`lrf`) | 0.01 |
| Warmup epochs | 3 |
| IoU | 0.7 |
| Box loss weight | 7.5 |
| Classification loss weight | 0.5 |
| HSV-S | 0.5 |
| HSV-V | 0.4 |
| Scale | 0.5 |
| Mosaic | 1.0 |
| Horizontal flip | 0.5 |
| Auto augmentation | RandAugment |
| Erasing | 0.4 |
| AMP | True |
| Optimizer | Auto |
| Pretrained | True |
| Class remapping | True |
| Seed | 0 |
| Deterministic | True |

The best weights from this run were used as the starting point for the final training stage.

### 10.4 Final model

| Parameter | Value |
|---|---:|
| Model | RT-DETR-L |
| Starting weights | Retraining 2 `best.pt` |
| Dataset | `datasets/retrain_03_09/data.yaml` |
| Epochs | 30 |
| Patience | 100 |
| Batch size | 4 |
| Image size | **600 × 600** |
| Cache | False |
| Workers | 2 |
| Close mosaic | 10 |
| Initial learning rate (`lr0`) | 0.01 |
| Final learning rate factor (`lrf`) | 0.01 |
| Warmup epochs | 3 |
| IoU | 0.7 |
| Box loss weight | 7.5 |
| Classification loss weight | 0.5 |
| HSV-S | 0.7 |
| HSV-V | 0.7 |
| Scale | 0.8 |
| Mosaic | 1.0 |
| Horizontal flip | 0.5 |
| Auto augmentation | RandAugment |
| Erasing | 0.4 |
| AMP | True |
| Optimizer | Auto |
| Seed | 0 |
| Deterministic | True |

The final production model is the resulting trained PyTorch `best.pt` file.

---

## 11. Experimental Validation

The evaluation was performed on the same independent evaluation dataset for the different experimental stages.

The reported metrics are:

- Precision
- Recall
- mAP@0.50
- mAP@0.50:0.95

### Metric definitions

**Precision**

```text
Precision = TP / (TP + FP)
```

Precision measures how many of the predicted detections are correct.

**Recall**

```text
Recall = TP / (TP + FN)
```

Recall measures how many of the ground-truth objects were detected.

Recall is particularly important in this project because missing a PPE violation is more important than producing an additional warning.

**mAP@0.50**

Mean Average Precision calculated at an IoU threshold of 0.50.

**mAP@0.50:0.95**

Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95.

---

## 12. Experimental Results

The four experimental stages were evaluated against the same independent evaluation dataset.

| Run | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|
| Baseline | 0.8933 | 0.7713 | 0.7500 | 0.3946 |
| Retraining 1 | 0.8117 | 0.8036 | 0.7591 | 0.3710 |
| Retraining 2 | 0.9126 | 0.7319 | 0.7130 | 0.4011 |
| **Final** | **0.9294** | **0.8618** | **0.8542** | **0.4522** |

The final model gave the best overall result among the four tested configurations.

---

## 13. Baseline vs Final Model

| Metric | Baseline | Final | Improvement |
|---|---:|---:|---:|
| Precision | 0.8933 | **0.9294** | +0.0361 |
| Recall | 0.7713 | **0.8618** | +0.0905 |
| mAP@0.50 | 0.7500 | **0.8542** | +0.1042 |
| mAP@0.50:0.95 | 0.3946 | **0.4522** | +0.0576 |

The final model therefore passed the proposal targets for mAP@0.50 and precision:

- mAP@0.50 target: **0.80** → final **0.8542**
- Precision target: **0.75** → final **0.9294**
- Recall: improved from **0.7713** to **0.8618**
- Inference FPS target: **25 FPS** → tested result **~6.5 FPS**

---

## 14. Per-Class Results

### Baseline

| Class | mAP@0.50 |
|---|---:|
| `Bare_Head` | 0.9217 |
| `No_Safety_vest` | 0.7693 |
| `Non_Compliant_Hat` | 0.6503 |
| `Safety_vest` | 0.5928 |
| `helmet` | 0.8162 |

### Retraining 1

| Class | mAP@0.50 |
|---|---:|
| `Bare_Head` | 0.9346 |
| `No_Safety_vest` | 0.8666 |
| `Non_Compliant_Hat` | 0.4456 |
| `Safety_vest` | 0.7385 |
| `helmet` | 0.8102 |

### Retraining 2

| Class | mAP@0.50 |
|---|---:|
| `Bare_Head` | 0.9350 |
| `No_Safety_vest` | 0.5480 |
| `Non_Compliant_Hat` | 0.6450 |
| `Safety_vest` | 0.6864 |
| `helmet` | 0.7506 |

### Final model

| Class | mAP@0.50 |
|---|---:|
| `Bare_Head` | **0.9350** |
| `No_Safety_vest` | **0.8551** |
| `Non_Compliant_Hat` | **0.8250** |
| `Safety_vest` | **0.8151** |
| `helmet` | **0.8410** |

The largest class-level improvements from baseline to final were for:

- `Safety_vest`: +0.2223
- `Non_Compliant_Hat`: +0.1747
- `No_Safety_vest`: +0.0858

---

## 15. Parameter-Space Analysis

The project uses the same RT-DETR-L model across the experiments and explores different training configurations rather than comparing different model architectures.

The four stages were used to investigate how changes in training configuration and dataset composition affected the same RT-DETR-L model.

The main changes across the experiments included:

- Training image resolution
- Number of epochs
- Batch size
- Dataset composition/balancing
- Patience
- Augmentation settings
- HSV augmentation
- Scale augmentation
- Mosaic closing point
- Final training dataset

The progression was:

```text
Baseline
   │
   ▼
Retraining 1
   │
   ▼
Retraining 2
   │
   ▼
Final model
```

### Important note about the analysis

The experiments were performed as an iterative development process rather than as a perfectly controlled one-variable-at-a-time scientific ablation.

Several training settings changed between some runs, especially when moving to the balanced retraining dataset. Therefore, the results should be interpreted as **iterative parameter-space exploration and configuration comparison**, not as proof that one individual parameter alone caused a specific improvement.

This limitation is reported explicitly rather than presenting the experiments as controlled single-parameter ablations.

The final model was selected because it gave the best overall evaluation result among the tested configurations and exceeded the main accuracy targets from the proposal.

---

## 16. Final Model Selection

The final model achieved:

```text
Precision       = 0.9294
Recall          = 0.8618
mAP@0.50        = 0.8542
mAP@0.50:0.95   = 0.4522
```

Compared with the baseline, the final model improved all four reported overall metrics.

The final model also improved the weaker classes that were limiting the baseline performance, especially `Safety_vest`, `Non_Compliant_Hat`, and `No_Safety_vest`.

---

## 17. Inference Speed

During testing of the final trained PyTorch weights, the model achieved approximately:

**6.5 FPS inference**

This corresponds to approximately:

```text
1 / 6.5 ≈ 0.154 seconds
```

or about **154 ms between inference checks**.

This is below the original 25 FPS target, so the project does not claim that the model achieved the proposed inference-speed target. However, the measured speed can still be useful for PPE compliance monitoring. The system is not trying to detect very fast events such as high-speed machinery motion. A worker does not normally take off a safety helmet or put it back on in a fraction of a second. At approximately 6.5 inference checks per second, the system is still checking the camera scene roughly every 154 ms.

The production pipeline also keeps the display loop separate from inference. During testing:

| Measurement | Result |
|---|---:|
| Model inference | **~6.5 FPS** |
| Time between inference checks | **~154 ms** |
| Display/UI loop | **~30 FPS** |
| Original proposed inference target | **≥25 FPS** |

The approximately 30 FPS value refers to the display loop, **not** the model inference speed.

The 6.5 FPS result is specific to the tested hardware/software configuration and is not a guaranteed speed for other computers.

---

## 18. Production Pipeline

The production system uses the trained PyTorch model and performs the following steps:

```text
Camera / RGB video
        │
        ▼
Frame capture
        │
        ▼
Resize to 640 × 640
        │
        ▼
BGR → RGB
        │
        ▼
Float32 normalization
        │
        ▼
RT-DETR-L best.pt
        │
        ▼
Detection results
        │
        ▼
PPE violation logic
        │
        ├───────────────┐
        ▼               ▼
Display          Violation detected
                        │
                ┌───────┴────────┐
                ▼                ▼
          Local alarm       Telegram snapshot
```

The production code uses Python threading so that video capture/display and inference do not have to run in the same blocking loop.

---

## 19. Alert System

### Local alarm

When a violation is detected, the system can trigger a local audio alarm.

The alarm file is:

```text
assets/alarm.wav
```

### Telegram notification

The Telegram functionality:

1. Captures the relevant frame.
2. Encodes the frame as JPEG.
3. Sends the image through the Telegram Bot API.
4. Applies a 5-second cooldown/debounce.

The Telegram implementation was tested during development using simulated dispatches and terminal output. The real Telegram notification path was implemented, but a real production Telegram delivery was not treated as a completed end-to-end test.

---

## 20. Installation

The production pipeline requires Python and the packages used by the project.

A basic environment can be installed with:

```bash
pip install ultralytics opencv-python requests
```

PyTorch is installed as required by the selected PyTorch/Ultralytics environment.

For a compatible GPU installation, the appropriate PyTorch build should be installed according to the local CUDA/hardware setup.

The tested development machine used:

- Intel Core i7 CPU
- NVIDIA GeForce GTX 1660 Ti, 6 GB VRAM
- Windows
- Python 3.11.x

The final production model is a PyTorch `.pt` model, so CUDA/PyTorch compatibility can affect inference speed.

---

## 21. Running the Production Pipeline

From the repository root:

```bash
python src/production_pipeline.py
```

The final model should be available at:

```text
weights/best.pt
```

The production preprocessing should use the final **640 × 640** input size.

---

## 22. Reproducibility

To reproduce the main production setup:

### Step 1 — Clone the repository

```bash
git clone https://github.com/smsafwan/SmartPPE_Safwan.git
cd SmartPPE_Safwan
```

### Step 2 — Install the required packages

```bash
pip install ultralytics opencv-python requests
```

Install the appropriate PyTorch version for the available hardware.

### Step 3 — Obtain the evaluation dataset

Use the evaluation dataset link:

```text
https://drive.google.com/file/d/1LpDP9M3984Qyh3PxFZ-m3ejX5ITx9mGy/view?usp=drive_link
```

### Step 4 — Place the final model

The production model should be:

```text
weights/best.pt
```

### Step 5 — Run

```bash
python src/production_pipeline.py
```

---

## 23. Sample Results

The repository should contain representative sample outputs.

### Compliant example

```text
assets/output_compliant.jpg
```

### Violation example

```text
assets/output-violation-1.jpg
assets/output-violation-2.jpg
assets/output-violation-3.jpg
```

These images are included to show the intended visual output of the system.

---

## 24. Existing Implementations and References

The project uses established computer vision software and the RT-DETR architecture rather than implementing the detector architecture from scratch.

### RT-DETR

The detection model is based on RT-DETR:

> Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, E., Liu, Y., & Chen, J. (2023). *DETRs Beat YOLOs on Real-time Object Detection.*

Paper:

https://arxiv.org/abs/2304.08069

The RT-DETR architecture provides an end-to-end object detection approach based on Transformers.

### Ultralytics

Ultralytics was used for:

- RT-DETR model training
- Transfer learning
- Training configuration
- Evaluation
- Model export/experimentation

Source:

https://github.com/ultralytics/ultralytics

### OpenCV

OpenCV was used for:

- Camera/video capture
- Image resizing
- Color conversion
- Frame handling
- Display

Source:

https://github.com/opencv/opencv

### PyTorch

PyTorch is used as the deep-learning framework behind the trained `.pt` model.

Source:

https://github.com/pytorch/pytorch

### Requests

The Python Requests library is used for HTTP communication with the Telegram API.

Source:

https://github.com/psf/requests

### Telegram Bot API

Telegram is used as the remote notification channel.

Documentation:

https://core.telegram.org/bots/api

---

## 25. What Was Reused

The project reuses:

- RT-DETR-L architecture
- Pre-trained RT-DETR weights
- Ultralytics training and evaluation functionality
- PyTorch
- OpenCV
- Requests
- Telegram Bot API

The model architecture and software libraries are existing technologies. They were adapted to the PPE monitoring problem rather than being implemented from scratch.

---

## 26. What Was Developed for This Project

The project-specific work includes:

- Definition of the five PPE detection classes
- Preparation of the training and retraining datasets
- Assembly of the independent evaluation dataset
- Manual ground-truth annotation
- Manual verification and correction of all annotated instances
- RT-DETR-L training
- Iterative retraining and parameter-space exploration
- Dataset balancing/retraining experiments
- Final model selection
- Independent evaluation against ground truth
- PPE violation logic
- Production video-processing pipeline
- Local alarm integration
- Telegram snapshot notification integration
- Alert cooldown/debounce
- Production testing and performance measurement
- Final project documentation

---

## 27. Third-Party Code and Licenses

The main third-party software used by the project is listed below.

| Software | Use in project | License / source |
|---|---|---|
| Ultralytics | RT-DETR training, evaluation and model tooling | AGPL-3.0 — https://github.com/ultralytics/ultralytics |
| PyTorch | Deep-learning framework and `.pt` model execution | BSD-3-Clause — https://github.com/pytorch/pytorch |
| OpenCV | Video and image processing | Apache 2.0 — https://github.com/opencv/opencv |
| Requests | Telegram HTTP requests | Apache 2.0 — https://github.com/psf/requests |
| Telegram Bot API | Remote notification service | https://core.telegram.org/bots/api |

The project should be used consistently with the licenses of the libraries and model tooling listed above.

---

## 28. AI / LLM Tools Declaration

AI-assisted development tools were used during the project for parts of the software development workflow.

In particular, **GitHub Copilot and Goggle Gemini** was used during the development stage for pipeline prototyping and coding assistance.

The AI tool was used for:

- Code suggestions
- Prototyping of parts of the production pipeline
- Development assistance during implementation

The project decisions, dataset preparation, model training, experimental evaluation, parameter selection, testing and final integration were performed by the project author.

---

## 29. Contributions

The project was developed by:

**Syed Muhammad Safwan**

### Project development

- Designed the overall SmartPPE system.
- Defined the PPE detection classes and evaluation workflow.
- Developed the training, evaluation and production workflow.
- Integrated the computer-vision model with the production pipeline.

### Dataset preparation

- Collected and assembled training/retraining data from multiple Roboflow datasets.
- Prepared the independent evaluation dataset from multiple YouTube video sources.
- Prepared the class mapping and dataset configurations.
- Personally inspected all annotated instances.
- Corrected incorrect bounding boxes, class assignments and missing/incorrect annotations.

### Model development

- Trained the initial RT-DETR-L baseline.
- Analysed baseline performance.
- Performed successive retraining experiments.
- Changed training resolution, batch size, training duration, dataset composition and augmentation settings during the experiments.
- Selected the final configuration based on the evaluation results.

### Evaluation

- Created and annotated the independent evaluation dataset from scratch.
- Evaluated the different experimental stages using the same evaluation dataset.
- Calculated and compared precision, recall, mAP@0.50 and mAP@0.50:0.95.
- Analysed per-class performance.
- Compared the final model against the baseline.

### Deployment

- Developed the production video-processing pipeline.
- Implemented asynchronous capture/inference/display processing.
- Integrated local audio warnings.
- Implemented Telegram snapshot notifications and the 5-second debounce logic.
- Tested the final trained model in the production pipeline.
- Measured inference and display performance.

### Documentation

- Prepared the repository structure.
- Documented the dataset, methodology, experiments, results and limitations.
- Prepared the final project documentation and presentation.

---

## 30. Limitations

The main limitations of the current system are:

1. **Inference speed:** the final PyTorch model achieved approximately 6.5 FPS on the tested setup, below the original ≥25 FPS target.
2. **Environment dependence:** performance can change with different hardware, drivers, camera resolution and software versions.
3. **Visible-light operation:** the system is intended for normal visible-light conditions and is not designed for complete darkness or IR-only operation.
4. **Dataset limitations:** the evaluation dataset contains 202 annotated object instances, so the reported results should not be treated as a complete representation of every possible industrial environment.
5. **Class imbalance:** the number of instances differs considerably between classes, especially in the retraining dataset.
6. **Telegram testing:** the Telegram alert mechanism was implemented and simulated during testing, but a complete real-world Telegram delivery test was not treated as a validated production result.
7. **No cross-camera tracking:** the system does not perform worker re-identification across separate cameras.
8. **No machine interlock:** the current project generates monitoring and alert outputs but does not directly control industrial machinery.

---

## 31. Final Result

The final RT-DETR-L model achieved:

| Metric | Final result |
|---|---:|
| Precision | **0.9294** |
| Recall | **0.8618** |
| mAP@0.50 | **0.8542** |
| mAP@0.50:0.95 | **0.4522** |
| Model inference speed | **~6.5 FPS** |
| Display/UI speed | **~30 FPS** |
| Input resolution | **640 × 640** |

The main accuracy targets from the proposal were achieved. The inference-speed target of 25 FPS was not achieved, but the final system still provides frequent PPE checks and a responsive display through asynchronous processing.

The project therefore demonstrates a working PPE monitoring prototype with an independently evaluated RT-DETR-L model and an integrated alert pipeline.

---

## 32. Project Status

**Status: Completed — Working Prototype**

The current project includes:

- Five-class PPE object detection
- Independently collected evaluation data
- Ground-truth annotations
- Baseline and iterative retraining experiments
- Final RT-DETR-L model
- Quantitative evaluation
- Real-time camera pipeline
- Local alarm
- Telegram notification functionality
- Asynchronous processing
- Documentation and reproducibility information

---
