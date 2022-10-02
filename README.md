# flow-DCEMRI
This is the code for paper "Pharmacokinetic Parameters’ Distribution Estimation with Normalizing Flow in Dynamic Contrast-Enhanced Magnetic Resonance Imaging"
## 1. config
edit config.toml
## 2. train
    train --name=flow --lr=0.0005 --gpu=0
## 3. test
    train --model_path=experiment/flow