# Introduction
This is the source code of our CVPR 2019 paper "Object-aware Aggregation with Bidirectional Temporal Graph for Video Captioning". Please cite the following paper if you use our code.

Junchao Zhang and Yuxin Peng, "Object-aware Aggregation with Bidirectional Temporal Graph for Video Captioning", 32nd IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8327–8336, 2019.

# Dependency
This code is implemented with tensorflow (1.13.1).

# Data Preparation
Here we take MSVD (Microsoft video description corpus) dataset as an example. For each video, 40 frames are extracted, and 5 objects are detected for each frame with Mask RCNN model. Then we use Resnet-200 model to extract features for frames and object images.

The bidirectional temporal graph can be found in 'features/msvd/SimilarityGraph'.

# Usage
Start training and tesing by executiving the following commands. This will train and test the model on MSVD datatset. 

    - sh run_msvd_train_forwardgraph.sh  ## train the model with forward temporal graph
    - sh run_msvd_train_backwardgraph.sh ## train the model with backward temporal graph
    
    - python msvd_main_fusion_test.py ## test the models with bidirectional temporal graph
    
For more information, please refer to our [CVPR paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Object-Aware_Aggregation_With_Bidirectional_Temporal_Graph_for_Video_Captioning_CVPR_2019_paper.html).

Welcome to our [Laboratory Homepage](http://www.wict.pku.edu.cn/mipl/home/) for more information about our papers, source codes, and datasets.

# Related repositories

[sam-tensorflow](https://github.com/HuiyunWang/sam-tensorflow)