# flow-DCEMRI
This is the code for paper K. Fang et al., "Normalizing Flow-based Distribution Estimation of Pharmacokinetic Parameters in Dynamic Contrast-Enhanced Magnetic Resonance Imaging," in IEEE Transactions on Biomedical Engineering, doi: 10.1109/TBME.2023.3318087.
https://ieeexplore.ieee.org/abstract/document/10258309
## 1. config
edit config.toml
## 2. train
    train --name=flow --lr=0.0005 --gpu=0
## 3. test
    train --model_path=experiment/flow
