# Image-Captioning
Image Captioning is the process of generating textual description of an image.It uses both Natural language Processing and 
Computer Vision to generate the captions.
The image information is encoded to a vector using a resnet50 model and decoded using LSTM's with a time distributed layer.

## Dataset:
https://www.kaggle.com/shadabhussain/flickr8k

This dataset consists of 8000 images which are further divided into train (6000 images), validation (1000 images) and test(1000 images) 
where each image consists of 5 captions with a similar meaning.

# Network Architecture:
![deconv_overall](./network.png)

## Encoder
The Convolutional Neural Network(CNN) can be thought of as an encoder. The input image is given to CNN to extract the features. 
The last hidden state of the CNN is connected to the Decoder.




