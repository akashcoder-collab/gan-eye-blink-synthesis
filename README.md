#  GAN Eye Blink Synthesis

##  Overview

This project implements a Generative Adversarial Network (GAN) to generate synthetic eye images for blink detection datasets.

##  Motivation

Blink detection datasets are limited and imbalanced. This project uses GANs to generate realistic eye images to improve dataset diversity.

##  Model Architecture

* **Generator**: ConvTranspose CNN (upsampling noise → image)
* **Discriminator**: CNN classifier (real vs fake)
* **Loss**: Binary Cross Entropy
* **Optimizer**: Adam (lr=0.0002, betas=(0.5, 0.999))

##  Training Details

* Image size: 64x64 (grayscale)
* Latent vector size: 128
* Epochs: 10
* Batch size: 128

##  Project Structure

```
src/
   dataset.py
   model.py
   train.py
   utils.py

data/
results/
models/
```

##  How to Run

```bash
pip install -r requirements.txt
python src/train.py
```

## Results

Synthetic eye images generated using GAN are saved in the `results/` folder.

##  Key Learnings

* Implemented GAN from scratch
* Understood adversarial training
* Learned synthetic data generation

##  Future Work

* Conditional GAN (open vs closed eyes)
* Improve image quality
* Real-time blink detection integration
