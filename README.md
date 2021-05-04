# Facial_Expression_Recognition
Final Project-Computational_Neuroscience (SYDE 552) 

### Problem Statement 

Picking up facial expression cues - what is an effortless task for humans is a surprisingly challenging task in the framework of computer vision. Facial expression recognition is a rather complex task for machines to learn and perform autonomously. In this work, we research on an automated facial expression recognition (FER) system, which in recent years has developed into a widely studied problem in the field of image processing. FER has applications in many interesting fields from robotics to biometrics.


### Summary 

The inspiration behind this work is to understand the implementation of Convolutional Neural Network (CNN)-based architectures, namely VGG on the Facial Expression Recognition (FER)-2013 dataset. Deep Learning is a large field of study and its application has found an important place in computer vision. From a high-level there is one method from deep learning that has received massive attention for application in computer vision - it is Convolutional Neural Networks (CNNs). One of the reasons being that being - CNNs are specifically designed for image data, such that development of an automated system can be attempted by a CNN for a facial expression recognition system. In the process we investigated varying topologies of VGG network, such as VGG1, VGG2, VGG3 from scratch and finally, VGG16 to investigate transfer learning. Ultimately, we aim to test and improve a CNN that aims to accomplish the task of recognition of seven basic human expressions: ‘anger’, ‘sad’, ‘happy’, ‘neutral’, ‘surprise’, ‘fear’, ‘disgust from a set of input images. It was found that images in the datasest were indeed very small, thus, it can be challenging to see what is represented in some of the images given the low resolution. This low resolution was likely the cause of the limited performance that top-of-the-line algorithms are able to achieve. To counter this, we can use images over 400 * 400 pixels.

### Results 

VGG3 (Dropout (0.2,0.3,0.4,0.5) + Augmentation + BatchNormalization) ---> Test Accruracy = 56.49% , Test Loss = 2.33

VGG16 (Transfer Learning) ---> Test Accuracy = 58.03%, Test Loss = 1.44 
![Figure_1](https://user-images.githubusercontent.com/38030229/117046339-9526d780-acde-11eb-94fe-a4241d8b5984.png)

### Data Availability 
The FER-2013 Dataset used in this paper is publicly available and downloadable from the following link: https://www.kaggle.com/msambare/fer2013

