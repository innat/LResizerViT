# Motivation 

In this notebook, it's shown how we can train a higher resolution image for vision transformer model. Usually transformer based models with high resolutoin may not fit into GPU. In such cases, we can adopt a **Trainable Resizer** mechanism as a backbone of the transformer models and perform as a joint  learning of the image resizer and recognition models.

[**Learning to Resize Images for Computer Vision Tasks** - Google Research](https://arxiv.org/pdf/2103.09950v1.pdf). For a given image resolutoin and a model, this research work answer how to best resize that image in a target resolutoin. Off-the-shelf image resizers for example: bilinear, bicubic methods are commonly used in most of the machine learning softwares. But this may limit the on-task performance of the trained model. In the above work, it's shown that typical linear resizer can be replaced with the **Learned Resizer**. Below is the overall proposed learnable resizer blocks.

![image resizer](https://user-images.githubusercontent.com/17668390/138250657-29995830-b903-447f-8729-09b72b90ab3c.png)

In the paper, they showed that the proposed resizer mechanism improve the classificaiton mdoels. The added the resizer mechanism to the classification mdoels such as `DenseNet`, `InceptionNet` etc. IN this way, we can input very image size and the resizer mechanism will downsample the image appropriately for the actual mdoel. 

![rtee](https://user-images.githubusercontent.com/17668390/138254072-f87daa13-12cc-4c6a-9145-a567f644cb12.png)

[**Vision Transformer** - Google Research](https://arxiv.org/pdf/2010.11929.pdf). We know that the transformer models are computationally expensive. And thus limits the input size roughly around `224`, `384`. So, the idea is to use this **resizer mechanism** as a bacbone of the **vision transformer**, so that we can input enough large image for **joint learning**. So, the overall model architecture would be 

![Presentation2](https://user-images.githubusercontent.com/17668390/138256285-c24f98db-ce35-4877-8741-221fd57d895e.jpg)

**Reference**

- [ROBIN SMITS](https://www.kaggle.com/rsmits/effnet-b2-feature-models-catboost#SET-TPU-/-GPU) - For general training pipelines. Great work. 
- [Learnable-Image-Resizing](https://github.com/sayakpaul/Learnable-Image-Resizing) For resizer building blocks. 
- [TensorFlow-HUB](https://github.com/sayakpaul/ViT-jax2tf) For ViT 


# Run 

[Notebook](https://www.kaggle.com/ipythonx/learning-to-resize-images-for-vision-transformer) 
