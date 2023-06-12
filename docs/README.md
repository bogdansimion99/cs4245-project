# Computer Vision in Tennis

### Group: 22
### Members: 

#### Alexandru Bobe: 5069831
#### Bogdan Simion: 5850185
#### Onno Verberne: 5883407

             
### This project was done in part to satisfy the requirements for the Seminar Computer Vision by Deep Learning course at TU Delft.
### We aim to reproduce and to improve some parts of the paper "TenniSet: A Dataset for Dense Fine-Grained Event Recognition, Localisation and Description". In this blog we’ll be elaborating on our efforts to reproduce the results, the issues we faced and the discussion about the possibility of reproducing the paper.

## Introduction

### Tennis, similar to every other field that has benefited from technology, has experienced significant advancements. Since Hawk-Eye developement [^1] many algorithms detecting the players' action, the score and tracking the ball were made. Some of the recent improvements in the field were encompassed by TrackNet [^2], which trains a deep learning network, addressing the problem of tracking fast moving tiny objects and Optical Flow [^3], a currently widely used technique for detecting moving objects. In terms of datasets, an important advancement was made by Faulkner and Dick [^4], who created the TenniSet, a dataset focused on event detection based on the players movements and ball's position. However, there are some flaws: the slow computation time makes it impossible to use these approaches for any real-time analysis of a tennis match. Furthermore, these techniques are not yet cheap and viable ways to create an annotated dataset that could be further used in research. Thus, we successfully some parts of the TenniSet paper as well as providing a new, faster approach for handling real-time tennis annotation. Last but not least, we would like to divide our discussion into several parts: the previous work related to annotation using computer vision in sports; our project goals and what we are trying to implement; the results we have for this setting and in the end, the discussion about the results and issues we faced.

## Previous work

### One of the first groundbreaking innovations in terms of technologies used is Hawk-Eye [^1]. The Hawk-eye tennis system is a cutting-edge technology designed to enhance decision-making in tennis matches. By utilizing advanced image processing techniques and sophisticated algorithms, Hawk-eye can accurately track the trajectory of the ball during gameplay. The system employs a network of high-speed cameras strategically positioned around the court to capture multiple angles of the ball's movement. The Hawk-eye technology has revolutionized the game of tennis by providing players, officials, and spectators with real-time, accurate information on ball placement. It has become a valuable tool in resolving disputed calls, as it allows officials to review and make more informed decisions on whether the ball landed in or out of bounds. The system's ability to provide quick and objective insights has greatly enhanced the fairness and integrity of the sport.

### The paper by Vinyes Mora [^5] presents a comprehensive framework for in-play tennis analysis using computer vision and machine learning techniques. This work builds upon previous research in the field of sports analysis, specifically focusing on real-time analysis of tennis matches. The author proposes a novel framework that combines computer vision algorithms and machine learning methodologies to extract meaningful insights from visual data obtained during live tennis matches. The paper contributes to the existing body of literature by presenting a detailed description of the framework and its underlying algorithms. It highlights the importance of computer vision techniques such as object detection, motion tracking, and player tracking, as well as the role of machine learning algorithms for classification and prediction tasks. The proposed framework offers an innovative approach to analyzing in-play tennis events, providing a deeper understanding of player movements, shot recognition, and other relevant aspects of the game.

### The paper by Huang et al. [^2] introduces TrackNet, a deep learning network specifically designed for tracking high-speed and small objects in sports applications. This work addresses the challenges associated with tracking objects such as balls or players in fast-paced sports scenarios, where objects can be both small in size and rapidly moving. The authors present TrackNet as a novel solution that combines the power of deep learning with specialized techniques for tracking objects in sports. The network architecture is designed to effectively handle the complexities of tracking high-speed objects by incorporating features such as motion prediction, object recognition, and temporal modeling.

### Faulkner et. al [^4] created a database of annotated tennis matches for automatic tennis commentary generation, called TenniSet. It is designed to encompass a wide range of tennis-specific events, enabling researchers and practitioners to develop and evaluate algorithms and models for event recognition and understanding. It includes detailed annotations for various events that occur during a tennis match, such as serves, volleys, forehands, backhands, and other distinct actions and movements. The dataset provides dense annotations, meaning that it offers precise temporal and spatial information about the occurrences of different events. This level of granularity allows for more nuanced and accurate event recognition and localization. Additionally, the Tenniset dataset includes textual descriptions of the events, providing valuable context and aiding in comprehensive event understanding.

## Project goals

### We aim to reproduce some parts of the Computer Vision algorithms presented in [^3], namely event detection and recognition as well as providing a new approach. To improve the performance of tennis annotation we propose adding an optical flow component to the network which will serve as a secondary input to the annotation network. To address the computational costs of optical flow calculations we propose two methods. The first one is frame interpolation, designed in "RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation", where we use a network to calculate the intermediate optical flow frames. The paper here uses network distillation to improve the performance of an existing optical flow frame interpolation network. The second one is network distillation, designed in "Craft Distillation: Layer-wise Convolutional Neural Network Distillation". Our goal is to distill an existing optical flow network layer by layer using Craft Distillation.

### In order to make sure that we were right about what is missing and that our proposed solution is feasible, we start by reproducing the results obtained by Faulkner et. al. Afterwards, we try to see if their solution generalizes to unseen videos of official tennis matches. To test our proposed solution, we use the same dataset and analyse our newly trained models from different perspectives including accuracy and training time.  

## Methodology

### ![Alt Text](https://github.com/bogdansimion99/bogdansimion99.github.io/blob/main/docs/assets/VGG16_Architecture.jpg)

## Results

###

## Issues
### 

## Discussion
### 

## References

### [^1]: Owens, N. E. I. L., Harris, C., & Stennett, C. (2003, July). Hawk-eye tennis system. In 2003 international conference on visual information engineering VIE 2003 (pp. 182-185). IET.
### [^2]: Huang, Y. C., Liao, I. N., Chen, C. H., İk, T. U., & Peng, W. C. (2019, September). TrackNet: A deep learning network for tracking high-speed and tiny objects in sports applications. In 2019 16th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS) (pp. 1-8). IEEE.
### [^3]: Dosovitskiy, A., Fischer, P., Ilg, E., Hausser, P., Hazirbas, C., Golkov, V., ... & Brox, T. (2015). Flownet: Learning optical flow with convolutional networks. In Proceedings of the IEEE international conference on computer vision (pp. 2758-2766).
### [^4]: Faulkner, H., & Dick, A. (2017, November). Tenniset: a dataset for dense fine-grained event recognition, localisation and description. In 2017 International Conference on Digital Image Computing: Techniques and Applications (DICTA) (pp. 1-8). IEEE.
### [^5]: Mora, Silvia Vinyes. Computer Vision and Machine Learning for In-Play Tennis Analysis: Framework, Algorithms and Implementation. Diss. Imperial College London, 2018.



