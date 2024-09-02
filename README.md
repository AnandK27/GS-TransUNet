# GS-TransUNet 🫲🏻❤️‍🩹🧑🏻‍⚕️
Anand Kumar, Kavinder Roghit Kanthen, Josna John <br>
[Manuscript](https://drive.google.com/file/d/1kkwTWt8kJJKMWFEPYQz2A0kuLONRSjqx/view)<br>
This is the official code for the paper "GS-TransUNet: Gaussian splatting skin lesion analysis"

## Abstract
  This research aims to develop a more effective and accurate automated diagnostic tool for skin cancer
analysis by simultaneously addressing lesion segmentation and classification tasks. Traditional methods typically
handle these tasks separately, which can lead to inefficiencies and reduced accuracy. The GS-TransUNet model
aims to integrate these tasks into a cohesive framework, leveraging advanced machine learning techniques to
improve diagnostic precision and speed.

By combining 2D Gaussian Splatting with Transformer UNet architecture, this study seeks to enhance the
consistency and accuracy of segmentation masks, which are crucial for the reliable classification of skin lesions. This
integrated approach improves diagnostic accuracy and reduces the computational cost associated
with separate processing stages, paving the way for real-time applications in clinical settings.

## Usage
### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

  1. Download the [ISIC-2017](https://challenge.isic-archive.com/landing/2017/) and [PH-2](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar)
  2. Make sure to change the root paths in [`dataset.py`](https://github.com/AnandK27/GS-TransUNet/blob/dsmlp_gauss/dataset.py) appropriately.


### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --xp_name gauss
```
Once training is done the script automatically runs test.

## Results (to be added)

## Reference
* [TransUNet](https://arxiv.org/pdf/2102.04306)
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
