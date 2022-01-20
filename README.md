# LResizerViT

An unofficial `TensorFlow 2` implementation of [Trainable **Resizer**](https://arxiv.org/pdf/2103.09950v1.pdf) and [**Vi**sion **T**ransformer](https://arxiv.org/pdf/2010.11929.pdf) joint training model (**LResizerViT**). Below is the overall proposed learnable resizer blocks.

![image resizer](https://user-images.githubusercontent.com/17668390/138250657-29995830-b903-447f-8729-09b72b90ab3c.png)



# Code Example:
- `dataloader/data.py` path contains a flower data set, 5 classificaiton task (demonstration purpose only). 
- Real Problem: [PetFinder.my - Pawpularity Contest](https://www.kaggle.com/c/petfinder-pawpularity-score) : 
[Solution Notebook](https://www.kaggle.com/ipythonx/learning-to-resize-images-for-vision-transformer) 


**Reference**

- [ROBIN SMITS](https://www.kaggle.com/rsmits/effnet-b2-feature-models-catboost#SET-TPU-/-GPU) - For general training pipelines. Great work. 
- [Learnable-Image-Resizing](https://github.com/sayakpaul/Learnable-Image-Resizing) For resizer building blocks. 
- [TensorFlow-HUB](https://github.com/sayakpaul/ViT-jax2tf) For ViT 
