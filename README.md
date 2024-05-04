Here's the updated README with the dataset information:

# ViTLoc: Transformer Approaches for Absolute Pose Regression

This repository contains three novel transformer-based approaches for camera-based localization and absolute pose regression from a single input image. The methods are designed to estimate the 6-DoF camera pose (position and orientation) without requiring additional modalities like GPS or LiDAR data.

## Datasets

The experiments and training are performed on the following datasets:

1. **Cambridge Landmarks Dataset**: This dataset consists of images of landmarks in Cambridge, UK, along with their corresponding camera poses.

2. **Oxford RobotCar Dataset**: This dataset contains images and poses captured by a vehicle driving around Oxford, UK.

## Methods

### 1. PoseFormer (Old School)

PoseFormer is a simple yet effective model that replaces the convolutional backbone in the classical PoseNet architecture with a Vision Transformer (ViT). It directly encodes the input image through the ViT's sequence of self-attention layers, enabling better capture of global context and long-range spatial relationships, improving pose estimation accuracy.

### 2. ThreeFormer (Sequential APR)

ThreeFormer is a multi-stream transformer model for multi-scene absolute pose regression. It employs dedicated position and orientation transformers that learn to emphasize different visual cues - corners/blobs for position encoding and edges/contours for orientation encoding. This specialized dual-attention strategy effectively aggregates pose-relevant features, while a transformer decoder combines the outputs into a unified pose prediction across multiple scenes.

### 3. FeatFormer (Geometry Based APR)

FeatFormer is an approach that combines transformers with 3D feature matching for robust pose estimation. It utilizes a ViT backbone to extract dense visual features from the input image, which are projected into a 3D feature volume through a neural radiance field (NeRF). Camera pose is then regressed by performing 3D-3D matching between the extracted features and a learned scene representation, optimized via a triplet loss on neighboring poses.

## Setup

Each method has its own codebase and conda environment file. Follow the steps below to set up the environment for each method:

### PoseFormer/ThreeFormer/FeatFormer

1. Navigate to the `PoseFormer` directory:
   ```
   cd PoseFormer
   ```

2. Create a new conda environment from the provided `conda.yaml` file:
   ```
   conda env create -f conda.yaml
   ```

3. Activate the environment:
   ```
   conda activate poseformer-env
   ```

4. You can now run the PoseFormer code.


* Please refer to the respective method directories for specific instructions on running the code and any additional dependencies or requirements.

## Training and Inference

### PoseFormer

#### Training

To train the PoseFormer model, run the following command:

```
python train.py --data-path /path/to/data --output-dir /path/to/output --epochs 50 --batch-size 32
```

This will train the PoseFormer model on the specified dataset for 50 epochs with a batch size of 32. You can adjust the hyperparameters according to your needs.

#### Inference

To perform inference using the trained PoseFormer model, run the following command:

```
python inference.py --model-path /path/to/model.pth --image-path /path/to/image.jpg
```

This will load the trained model and estimate the camera pose for the given input image.

### ThreeFormer

#### Training

To train the ThreeFormer model, run the following command:

```
python train.py --data-path /path/to/data --output-dir /path/to/output --epochs 100 --batch-size 64 --multi-scene
```

This will train the ThreeFormer model on the specified dataset for 100 epochs with a batch size of 64, considering multiple scenes. You can adjust the hyperparameters according to your needs.

#### Inference

To perform inference using the trained ThreeFormer model, run the following command:

```
python inference.py --model-path /path/to/model.pth --image-path /path/to/image.jpg
```

This will load the trained model and estimate the camera pose for the given input image across multiple scenes.

### FeatFormer

#### Training

To train the FeatFormer model, you need to follow a two-stage process:

1. Train the FeatFormer network and the histogram-assisted NeRF:

```
python train_stage1.py --data-path /path/to/data --output-dir /path/to/output --epochs 50 --batch-size 16
```

2. Train the direct feature matching module:

```
python train_stage2.py --data-path /path/to/data --output-dir /path/to/output --epochs 25 --batch-size 8
```

You can adjust the hyperparameters according to your needs.

#### Inference

To perform inference using the trained FeatFormer model, run the following command:

```
python inference.py --model-path /path/to/model.pth --image-path /path/to/image.jpg
```

This will load the trained model and estimate the camera pose for the given input image using the direct feature matching approach.

Please note that the provided command lines are placeholders, and you may need to modify them based on the actual implementation and configurations of each method. Additionally, make sure to replace `/path/to/data`, `/path/to/output`, `/path/to/model.pth`, and `/path/to/image.jpg` with the appropriate paths in your system.