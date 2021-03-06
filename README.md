# FuzzyAdventure

This repository consists of some explorational work done on LeNet and DCGAN. 
**More details for each section can be found in the README files of the folders themselves.**

## Files

### i) 4.1 SNN Model 
  * The markdown file contains a systematic approach to modelling a deep convolutional SNN.
  * The Images folder contains Reference images and comparisons to past work.
  
### ii) 4.2.1 LeNet CV
  * Contains the main notebook LeNet_75%, in which I have trained on the CIFAR10 dataset.
  * Contains the DCGAN notebook, in which I have trained an adversarial net on the CIFAR10 dataset.
  * Contains the EfNet-B0 notebook, in which I retrained a pretrained model (EfficientNet-B0) on the CIFAR10 dataset, to label images generated by the GAN generator.
  * Contains the MNIST notebook, in which I tested data augmentation techniques using the MNIST dataset.
  * Contains Data.zip, which has 1500(due to size limitations) of the 20000 images generated by the GAN.
  * Contains a GAN images folder which shows the progression of the Generated images at 8 different times during training.
  * Contains the benchmarking results for the numerous tests I performed on LeNet_75% training. 
  * Lastly, for reproducibility, I have kept manual seeds, and all the .pth files can also be found in the folder.
  
### iii) 4.2.2 NLP Reddit Flair Detector
  * Contains the main notebook NLP_Reddit, in which I have implemented the RNN Network (LSTM) and it's results.  
