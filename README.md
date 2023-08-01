# CT Scan Organ Localization With Reinforcement Learning

## Project Goal

We want to design a reinforcement learning algorithm that can automatically localize specific organ(s) in CT images to improve the working efficiency of the disease diagnosis progress.
- Baseline: a reinforcement learning-based organ localization model proposed by Navarro et al. (2020) 
- Goal: train the model with relatively small number of labeled data using RL-based model and try to improve the model performance
- Evaluation Metrics: 3D IoU score (intersection over Union)

## Dataset
We used a publicly available ![Multi-Atlas Labeling CT Dataset](https://doi.org/10.7303/syn3193805), which contains 30 abdomen CT scans with labels of 13 different organs.

## Example Liver Localization with Train RL Model
Axial View             |  Coronal View | Sagittal View 
:-------------------------:|:-------------------------:|:-------------------------:
  <img src="https://github.com/ykxxx/CT-Organ-Localization-With-Reinforcement-Learning/blob/main/image/liver-localization-example1.gif" width="98%"/>  |    <img src="https://github.com/ykxxx/CT-Organ-Localization-With-Reinforcement-Learning/blob/main/image/liver-localization-example2.gif" width="95%"/> | <img src="https://github.com/ykxxx/CT-Organ-Localization-With-Reinforcement-Learning/blob/main/image/liver-localization-example3.gif" width="95%"/> 
<!--   <figcaption>{{ Axial View }}</figcaption> -->
<!--   <figcaption>{{ Coronal View }}</figcaption> -->
<!--   <figcaption>{{ Sagittal View }}</figcaption> -->

## Model Framework
We optimized the RL model from the baseline paper and made several improvements to get the following architecture: 
|                      | **Baseline Model**                                                 | **Our Model**                                                                                                                            |
|----------------------|--------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Rewards Function     | Binary Rewards: -1, +1                                             | Proportional to change in IoU score Additional terminal rewards Penalty for out-of-boundary action                                       |
| Terminal Condition   | Oscillation occurrence  Or reaches terminate IoU: fixed to be 0.85 | Max step size, increased with more training epochs: 15 to 20 Or reaches terminate IoU, increased with more training epochs: 0.60 to 0.85 |
| Action Update Method | Proportional to bounding box length                                | Proportional to bounding box length Set minimum and maximum threshold Handle out-of-boundary cases                                       |
<img src="https://github.com/ykxxx/CT-Organ-Localization-With-Reinforcement-Learning/blob/main/image/model%20framework.png">
