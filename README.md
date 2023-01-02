# Project we made as a group for the Course CS F425 - Deep Learning.

## Introduction
Image Captioning is the process of generating textual descriptions of an image. It uses NLP and Computer 
Vision to generate the captions for the images. The datasets suitable for this problem consist of input 
images and their corresponding output captions. An encoder-decoder network is generally preferred to 
build an Image Captioning System. CNNs (Convolutional Neural Networks), which can be thought of as 
encoders, are used to extract the features from the Image, and RNNs (Recurrent Neural Networks), which 
can be thought of as decoders, are used to perform language modeling up to the word level. - phase architecture to build and train an Image Caption Generator. 
The link for the paper - https://drive.google.com/file/d/15XVNL76j3E8vQfmUSVvoqGu90O4h1-qm/view
We have made use of the Flickr 8k dataset, which consists of approximately 8000 sample images along 
with 5 captions with each image, thus giving about 40,000 captions in total. For the CNN Part to encode 
representations from the input images, we have used the VGG16 (Visual Geometry Group) pre-trained 
CNN model, which is a 16 convolutional layer model used for object recognition. For the RNN part, to learn 
the representations from the textual captions, we have used both Long short-term memory (LSTM) and 
Gated recurrent unit (GRU), and we have compared the performances of both these RNN Layers.

## Model Architecture
As mentioned, we have used a three-phase architecture to build our Image Captioning Generator.
1. Feature Extraction Phase: In this phase, the primary goal is to extract features from an image for 
training. 
 This phase first uses a VGG-16 architecture to extract features from the images using a 
combination of 3*3 convolutional layers and max-pooling layers. 
 The output of this architecture is passed to a dropout layer with a dropout rate of 0.5. This layer is 
added to reduce overfitting in the feature extraction phase. -Units) activation function on the extracted features. 
 The final output of this phase is features extracted in the form of vectors of size 256. 
2. Encoder Phase: 
 The captions are first tokenized and converted to numerical form, all using a tokenizer. These 
tokenized captions are padded so that the length of each vector generated is equal to the length of 
the longest tokenized caption. 
 These tokenized vectors are transformed to an output space of 256 by 31 (maximum length of 
the captions). 
 These vectors passed to a dropout layer with a dropout rate of 0.5 to reduce overfitting in this 
phase. 
 Then, the RNN layer is introduced, which helps the model learn how to generate valid sentences 
by generating the words with the highest probability of occurrence. 
 In our project, we individually used LSTM and GRU as our RNN layer. 
 The final output of this phase is features extracted in the form of vectors of size 256.
3. Decoder Phase: 
 The primary goal of the decoder phase is to concatenate the outputs of the Feature Extraction 
Phase and Decoder Phase. This phase also produces the required output, which is the predicted 
word given an image and the caption generated till that point in time. -
function and a small regularization parameter. 
 Another dense layer is added with the vocabulary s SoftMax
the activation function to generate the word in the vocabulary with the most probability of being 
placed next in the caption. 

![image](https://user-images.githubusercontent.com/84556711/210264517-58f620b6-e9ae-4828-b7a1-db44cc266785.png)


## PROPOSED CHANGES / INNOVATIONS
To improve the accuracy (BLEU Scores) of our model and reduce the validation losses, we have modified 
the model architecture by replacing the VGG16 model in the Feature Extraction Phase with the Inceptionv3 
module, a pre-trained model.
The Inceptionv3 is a much deeper network compared to VGG16, as it contains 42 layers in total compared to VGG16's 16 conv layers . Despite such a deep network, Inception v3 consists of only about 25 million parameters compared to VGG16's 135 million parameters.

These differences help the Inceptionv3 module train much faster than the VGG16 network and also learn better 
features and representations of the input data owing to its deeper architecture. This motivated us to implement 
the Inceptionv3 module in place of VGG16 to improve the performance of our model.
On the RNN side, we again individually used both LSTM and GRU and compared the performances

The three-phase architecture and the training procedure remain the same as the original model. In terms of 
dimensions, the only change occurs in the output dimension of the CNN network, where Inceptionv3 returns vectors of size 2048 inplace of VGG16's 4096 dimensional vectors.

![image](https://user-images.githubusercontent.com/84556711/210264811-f5288ee2-9be6-4dce-a881-7eab4a5df743.png)


## RESULTS FROM PAPER IMPLEMENTATION

The paper implemented by us uses two architectures:
1. VGG as the CNN Architecture and LSTM as the RNN Architecture 
2. VGG as the CNN Architecture and GRU as the RNN Architecture 
VGG-LSTM Architecture:
The Validation Error is found to be 3.74316
The Training Error is Found to be 3.2001 
VGG-GRU Architecture:
The Validation Error is found to be 3.73906
The Training Error is Found to be 3.2832

**Training Time**
The time taken to train the LSTM architecture is around 190-200s per epoch which is greater than 
that of the GRU architecture (140 - 150s per epoch) which is consistent with the findings of the 
paper.
 This occurs due to the comparatively higher no. of operations occurring in LSTM as compared to 
GRU.<br/>
**BLEU Score Analysis**
BLEU Score also called as Bilingual Evaluation Understudy Score is used as measure for 
 We calculated four types of BLEU Scores for both the training and testing data - 
BLEU -1 (1.0, 0, 0, 0), BLEU -2 (0.5, 0.5, 0, 0), BLEU -3 (0.33, 0.33, 0.33, 0) and 
BLEU -4 (0.25, 0.25, 0.25, 0.25). 
 We have used the cumulative weights since they give better output.
 
 
 ## RESULTS FROM PROPOSED CHANGES / INNOVATIONS
As a part of improvement for the Paper we have used InceptionV3 as CNN architecture for reasons 
discussed before and employed it in combination with both LSTM and GRU Architectures.
Inception-LSTM Architecture: 
The Validation Error is found to be 3.66195
The Training Error is Found to be 3.1079
Inception-GRU Architecture:
The Validation Error is found to be 3.68882
The Training Error is Found to be 3.1757

