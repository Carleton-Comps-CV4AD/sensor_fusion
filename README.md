# Sensor Fusion for Semantic Segmentation

This is a extention of MIT's ADE20K implementations of semantic segmentation algorithms to include a variety of sensor fusion variants.

## Creating the environment:

Create a conda environment with Python 3.11.11

Install the packages specified in requirements.txt:

conda create --name [name] --file requirements.txt

## Navigating the codebase

The changes include a few copies of dataset.py, train.py, and eval_multipro.py with adaptations for various architectures. 

dataset.py : Original implmementation from MIT.

dataset_1_channel.py : Depth only dataloader. Normalization constants are configured based on empirical results from depth dataset.

dataset_4_channel.py : RGBD dataloader. Includes base RGB normalization constants and our values for depth as well.

train_base.py : Original implementation from MIT.

train.py : Used primarily for prototyping new architures. Adjust the imported dataset module as needed to trial different implementations.

train_parallel.py : Used to train mid fusion model, with two parallel encoders.

eval_multipro_base.py : Original implementation from MIT.

eval_multipro.py : Used for prototyping. Adjust inplanes and slicing of image data for visualization as appropriate.

eval_multipro_parallel.py : Used to evaluate the mid fusion model.

Model changes are included in models.py and include:
 - Parallel encoder segmentation module (for mid fusion)
 - SE block (not currently used in the model, but easily added)
 - 3 layer decoder (deep supervision or not)
 - 5 layer decoder (deep supervision or not)

The HRnet implementation also has some small changes to allow for adjusting of inplanes. This is the only encoder supported by the fusion implementations.

## Usage:

Usage varies slightly by exact version of train or eval_multipro, but for the most part follows this pattern:

python3 train.py --cfg [relative path to config file] --gpus [0 for 1 gpu. not tested for multiple]

python3 eval_multipro.py --cfg [relative path to config file] --gpus [0 for 1 gpu. not tested for multiple] --test_set [relative path to folder containing test.odgt.optional, overrides the test side provided in the config] 

Examples for a variety of configuration files are provided in ./config. Note that some of them, such as lidar_only and random_noise either require some adjustment of the trainer and models, and are included for completeness sake. 

## Scripts:

graphs.ipynb : result visualization
count_class_instances.ipynb : used to explore class bias in the datasets. counts number of images which contain each class, not total instances of a class.
interpolation_prototyping.ipynb : prototyping and visualization of depth upsampling method + work in progress on incorporating entropy into the fusion model and visualizing the feature maps.
normalize.ipynb : used to obtain normalization constants for depth data and check that normalization is working correctly.
stack_lidar_rgb.ipynb : protyping for early fusion (rgbd with one encoder).

# Semantic Segmentation on CARLA Data
# Adapted from MIT ADE20K dataset in PyTorch

This is a PyTorch implementation of semantic segmentation models on custom data from the CARLA simulator. The implementation is adapted from the MIT ADE20K scene parsing dataset (http://sceneparsing.csail.mit.edu/).


## Background of MIT ADE20K
ADE20K is the largest open source dataset for semantic segmentation and scene parsing, released by MIT Computer Vision team. Follow the link below to find the repository for their dataset and implementations on Caffe and Torch7:
https://github.com/CSAILVision/sceneparsing


### ADE20K Methods on Syncronized Batch Normalization on PyTorch
This module computes the mean and standard-deviation across all devices during training. We empirically find that a reasonable large batch size is important for segmentation. We thank [Jiayuan Mao](http://vccy.xyz/) for his kind contributions, please refer to [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) for details.



### State-of-the-Art models
- **PSPNet** is scene parsing network that aggregates global representation with Pyramid Pooling Module (PPM). It is the winner model of ILSVRC'16 MIT Scene Parsing Challenge. Please refer to [https://arxiv.org/abs/1612.01105](https://arxiv.org/abs/1612.01105) for details.
- **UPerNet** is a model based on Feature Pyramid Network (FPN) and Pyramid Pooling Module (PPM). It doesn't need dilated convolution, an operator that is time-and-memory consuming. *Without bells and whistles*, it is comparable or even better compared with PSPNet, while requiring much shorter training time and less GPU memory. Please refer to [https://arxiv.org/abs/1807.10221](https://arxiv.org/abs/1807.10221) for details.
- **HRNet** is a recently proposed model that retains high resolution representations throughout the model, without the traditional bottleneck design. It achieves the SOTA performance on a series of pixel labeling tasks. Please refer to [https://arxiv.org/abs/1904.04514](https://arxiv.org/abs/1904.04514) for details.


## Models

We trained the model on CARLA data with a MobileNetV2dilated encoder. This was adpated from the pretrained models provided in the ADE20K paper. 

Encoder:
- MobileNetV2dilated


Decoders:
- C1 (one convolution module)
- C1_deepsup (C1 + deep supervision trick)
- PPM (Pyramid Pooling Module, see [PSPNet](https://hszhao.github.io/projects/pspnet) paper for details.)
- PPM_deepsup (PPM + deep supervision trick)




## Reference

Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso and A. Torralba. International Journal on Computer Vision (IJCV), 2018. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2018semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Xiao, Tete and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={International Journal on Computer Vision},
      year={2018}
    }

Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }
    
