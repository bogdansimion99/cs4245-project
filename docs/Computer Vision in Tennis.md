# Computer Vision in Tennis

### Group: 22
### Members: 

#### Alexandru Bobe: 5069831
#### Bogdan Simion: 5850185
#### Onno Verberne: 5883407
             

#### This project was done in part to satisfy the requirements for the Seminar Computer Vision by Deep Learning course at TU Delft.
----

We aim to reproduce and to improve some parts of the paper "TenniSet: A Dataset for Dense Fine-Grained Event Recognition, Localisation and Description". In this blog we’ll be elaborating on our efforts to reproduce the results, the issues we faced and the discussion about possible future work that builds on it.

## Table of Contents  
**[Introduction](#Introduction)**<br>
**[Previous work](#Previous-work)**<br>
**[Project goals](#Project-goals)**<br>
**[Methodology](#Methodology)**<br>
**[Results](#Results)**<br>
**[Issues](#Issues)**<br>
**[Discussions](#Discussions)**<br>

## Introduction


In the world of sports, technological advancements have brought about remarkable changes, captivating players and spectators alike. From the introduction of Video Assistant Referee (VAR) in football to the groundbreaking Hawk-Eye system in tennis, the possibilities for leveraging technology in sports seem boundless. These advancements have opened doors to exciting applications that benefit players, coaches, and enthusiasts. In particular, tennis has witnessed the emergence of Hawk-Eye, a computer vision system that automates refereeing by precisely tracking the ball's trajectory using an array of sophisticated cameras. On top of its immediate use to help umpires make better decisions, the system also creates a huge and up-to-date dataset that can improve the game in multiple directions.  However, despite the potential for this data to bridge the gap between players of different economic backgrounds, they have, unfortunately, only served to magnify it, in the sense that the richest players in the world afford to analyse it and get an almost unfair advantage against emerging players.

![Federer's Serve Analysis](./assets/fede.png)

Therefore, our goal is to explore the open-source research related to Computer Vision in tennis, understand its shortcomings and replicate the most promising work while keeping the idea of economic accessibility in mind throughout the whole project.

## Previous work

From the outset of our project, we recognized that the scope of our work would heavily rely on acquiring timely and relevant data. As a result, our initial literature review focused specifically on papers that provided publicly available data, in order to identify research that could be reproduced. This process led us to a disheartening realization: open-source tennis research has stagnated due to a profound lack of high-quality data accessible for scholarly investigation. It became apparent that the complexity of projects undertaken by commercial entities far surpassed the capabilities of the open-source research community. Supporting our hypothesis, Mora [^5] eloquently elaborates on this issue in her paper, shedding light on the scarcity of data for open-source tennis research and the challenges posed by its low quality and limited scale. This compelling evidence solidifies the urgent need to address the data deficit in order to propel open-source tennis research forward.

The contribution of Mora [^5] to the field of Computer Vision applied to tennis is significant. She presents a comprehensive framework for in-play tennis analysis using computer vision for object detection, motion tracking, and player tracking. Using these techniques, the system offered an innovative approach to analyzing in-play tennis events, providing a deeper understanding of player movements, shot recognition, and other relevant aspects of the game. Afterwards, we explored how we can also achieve the same results and maybe even improve them by using the more recent developments in the field.

<img src="./assets/tracking.png" width="640" height="360" title="tracking"/>

The first logical step was to understand how to track the ball. The paper by Huang et al. [^2] introduces TrackNet, a deep learning network specifically designed for tracking high-speed and small objects in sports applications. This work addresses the challenges associated with tracking objects such as balls or players in fast-paced sports scenarios, where objects can be both small in size and rapidly moving.

The second step was to find how we can track the players. Of course, all the iterations of the YOLO [^6] model were good candidates for our system, but we wanted to check if there are more task-specific models. While looking for a better alternative, we found a completely different solution proposed by Faulkner et al. [^4]. Instead of tracking the players and the ball with bounding boxes and later analyse their positions for getting insights, Faulkner *skipped* the step of tracking and went directly to action detection by analysing the frames of the video. Using this technique, they were able to perform frame classification, event detection and recognition and automatic commentary generation. Moreover, compared to most other papers, the dataset was publicly available and it was possible to reproduce the results.   



<!-- Information on optical flow models -->



## Methodology

In this section, we described our dataset together with the preprocessing steps, the splitting and the sampling technique used. Afterwards, we describe the models along with some implementation details. 
<!-- I think we also need to name the real time performance we are looking for somewhere in the intro -->
<!-- I wasn't sure what the end story would be. I put it for now that we were looking for "economical accesibility". -->
### Dataset
The dataset created by Faulkner et al. is based on videos of official tennis matches. The dataset contains 5 videos of full tennis matches, corresponding to more than 200GB of frames. Given our computation power limitation, we chose to use only a single video, which resulted in around 80k frames.
The frames are part of 11 classes, meaning 
* the first letter describing: Serve or Hit, 
* second letter: Far or Near, depending where the player is positioned in the frame 
* last letter: has multiple options, explained in the figure:
 
 ![Classes](./assets/tennis_cls.svg)


<p float="left">
  <img src="./assets/1783.png" width="220" title="OTH" />
</p>

<p float="left">
  <img src="./assets/11161.png" width="220" title="SFI"/> 
  <img src="./assets/10871.png" width="220" title="SFF"/>
  <img src="./assets/13024.png" width="220" title="SFL"/>
  <img src="./assets/1967.png" width="220" title="SNI"/> 
  <img src="./assets/6815.png" width="220" title="SNF"/>
  <img src="./assets/30021.png" width="220" title="SNL"/>
</p>

<p float="left">
  <img src="./assets/2777.png" width="220" title="HFL"/> 
  <img src="./assets/2004.png" width="220" title="HFR"/>
  <img src="./assets/2090.png" width="220" title="HNL"/>
  <img src="./assets/2025.png" width="220" title="HNR"/> 
</p>


### VGG16



<!-- ### VGG16

#### The VGG16 architecture consists of a series of convolutional layers, followed by fully connected layers. It is named "VGG16" because it has 16 weight layers, including 13 convolutional layers and 3 fully connected layers. The convolutional layers are designed to extract hierarchical features from the input images, while the fully connected layers act as a classifier to predict the class labels. -->

<!-- #### Each convolutional layer in VGG16 applies a set of learnable filters to the input image. These filters capture different aspects of the image, such as edges, textures, and shapes. The filters are small in spatial dimension but extend across the full depth of the input volume, enabling the model to learn rich spatial representations. -->

<!-- #### In VGG16, the convolutional layers are stacked on top of each other, with occasional max pooling layers in between. The max pooling layers downsample the spatial dimensions of the feature maps, reducing the computational complexity and increasing the receptive field of the subsequent layers. -->

<!-- #### The fully connected layers in VGG16 take the output of the last convolutional layer, flatten it into a 1-dimensional vector, and process it through a series of densely connected layers. The final fully connected layer produces the output predictions by employing a softmax activation function, which assigns probabilities to each class label. -->

<!-- #### During the training phase, VGG16 is typically trained using the backpropagation algorithm with gradient descent optimization. The weights of the network are updated iteratively to minimize a loss function, such as categorical cross-entropy, by comparing the predicted probabilities with the ground truth labels. -->

<!-- ![VGG16 Architecture](./assets/VGG16_Architecture.png) -->

<!-- ### Optical Flow

#### Optical flow refers to the pattern of apparent motion of objects in a sequence of images or video frames. It provides valuable information about the movement of objects and can be used for various computer vision tasks, such as object tracking, motion analysis, and video stabilization. To compute optical flow, we leverage the assumption that pixel intensities of objects in consecutive frames tend to remain constant unless affected by motion. Based on this assumption, several algorithms have been developed to estimate the motion vectors of pixels between frames. -->

<!-- #### More recently, deep learning-based methods have been developed to estimate optical flow. These approaches utilize convolutional neural networks (CNNs) to learn complex motion patterns and capture long-range dependencies. Networks like FlowNet and PWC-Net have achieved state-of-the-art performance in optical flow estimation by training on large-scale annotated datasets. In our study, we use optical flow as a key component in our methodology to analyze and track the motion of players and tennis balls in our tennis dataset. -->

<!-- ![Optical Flow Architecture](./assets/Optical_Flow_Architecture.png) -->

### Optical flow
In the 2017 paper by Faulkner et al. they demonstrate that the inclusion of optical flow data increases their models performance. The inclusion of motion information seems logically and empirically important for clasification of tennis videos. However, the optical flow model FlowNet [^7] dates back to 2015 and is quite slow, not ideal for real-time calculation. Over the years much more efficient models have been created, namely PWC-net [^8] by Nvidia and RAFT [^9] are two strong competitors. We have opted to use the latter RAFT as there is an existing easy to use pretrained pytorch implementation.

#### Two-Stream
In addition to the pure optical flow model, Faulkner et al. also proposed a two-stream model, where a standard VGG16 model and an optical flow VGG16 model have their features joined as they are passed into the classifier part of VGG16. This model saw the the greatest performance across the board.

### Network Distillation
TODO: Big cumbersome network, make it smaller via student teacher or craft distillation

## Results

#### a

## Issues
TODO: comparatively low computation power, runs taking very very long, lots of data so online is harder, running out of memory issues too
#### Imbalanced classes

## Discussion

#### a

## References

[^1]: Owens, N. E. I. L., Harris, C., & Stennett, C. (2003, July). Hawk-eye tennis system. In 2003 international conference on visual information engineering VIE 2003 (pp. 182-185). IET.
[^2]: Huang, Y. C., Liao, I. N., Chen, C. H., İk, T. U., & Peng, W. C. (2019, September). TrackNet: A deep learning network for tracking high-speed and tiny objects in sports applications. In 2019 16th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS) (pp. 1-8). IEEE.
[^3]: Dosovitskiy, A., Fischer, P., Ilg, E., Hausser, P., Hazirbas, C., Golkov, V., ... & Brox, T. (2015). Flownet: Learning optical flow with convolutional networks. In Proceedings of the IEEE international conference on computer vision (pp. 2758-2766).
[^4]: Faulkner, H., & Dick, A. (2017, November). Tenniset: a dataset for dense fine-grained event recognition, localisation and description. In 2017 International Conference on Digital Image Computing: Techniques and Applications (DICTA) (pp. 1-8). IEEE.
[^5]: Mora, Silvia Vinyes. Computer Vision and Machine Learning for In-Play Tennis Analysis: Framework, Algorithms and Implementation. Diss. Imperial College London, 2018.
[^6]: J. Redmon, S. Divvala, R. Girshick and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 779-788, doi: 10.1109/CVPR.2016.91.
[^7]: Dosovitskiy, A., Fischer, P., Ilg, E., Häusser, P., Hazirbas, C., Golkov, V., van der Smagt, P., Cremers, D., & Brox, T. (2015). FlowNet: Learning Optical Flow with Convolutional Networks. 2015 IEEE International Conference on Computer Vision (ICCV), 2758-2766.
[^8]: Sun, D., Yang, X., Liu, M., & Kautz, J. (2017). PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8934-8943.
[^9]: Teed, Z., & Deng, J. (2020). RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. European Conference on Computer Vision.
