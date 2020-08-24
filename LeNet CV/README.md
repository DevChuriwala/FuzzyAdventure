# CV Task - CIFAR10 using LeNet Architecture

## Final Validation Accuracy : 75.74%


### i) LeNet Classifier for CIFAR10
  * Initial Model trained using 2 conv layer containing 20, 50 filters respectively with a kernel size of 5 and stride of 1.
  * The fully connected layer contains 120 nodes, which I decided upon given the fact that CIFAR10 should have a greater number of distinct features in the data as compared to the MNIST dataset, which LeNet was made for.
  * **Baseline Accuracy of the Model : 65.82%**
  * Results can be found [here](https://github.com/DevChuriwala/FuzzyAdventure/blob/master/LeNet%20CV/Benchmarking%20Results/Baseline.png).

### ii) DCGAN on  CIFAR10
  * This is an implementation of the pre-existing DCGAN model.
  * I carefully tuned the hyperparameters to balance the discriminator and generator, since our final goal is to generate decent images to augment the CIFAR10 dataset.
  * The hyperparameters used are as follows: Epochs = 10; Batch Size = 64; LR = 2e-4; Beta1(for Adam) = 0.5; Beta2(for Adam) = 0.999.
  * The notebook can be found [here](https://github.com/DevChuriwala/FuzzyAdventure/blob/master/LeNet%20CV/DCGAN.ipynb).
  * The progression of the images can be found [here](https://github.com/DevChuriwala/FuzzyAdventure/tree/master/LeNet%20CV/GAN%20Images).
  * 1500 of the 20000 generated images can be found [here](https://github.com/DevChuriwala/FuzzyAdventure/blob/master/LeNet%20CV/Data.zip).
  * To label the generated data, I could not find a good pretrained model that used PyTorch, so I trained my own, based on EfficientNet-B0 and achieved an accuracy of **84.23% on CIFAR10**. I used this model to label my generated images. The notebook for which can be found [here](https://github.com/DevChuriwala/FuzzyAdventure/blob/master/LeNet%20CV/EfNetB0_CIFAR10.ipynb).
  
### iii) LeNet Classifier for MNIST
  * I trained the same network that I created for CIFAR10 on the MNIST dataset. Overfitting was carefully avoided using common generalization and regularization techniques.
  * Initially the model achieved an accuracy of **99.41%** without any optimization or augmentation. Results for which can be found [here](https://github.com/DevChuriwala/FuzzyAdventure/blob/master/LeNet%20CV/Benchmarking%20Results/MNIST_OG.png).
  
  
### Bonus Benchmarking Tasks (i) and (ii):
I benchmarked my baseline model using various augmentation and optimization techniques on CIFAR10 and MNIST. All the tests were run for 50 epochs with everything constant, except the technique being tested. As for MNIST the highest accuracy that I was able to achieve was ~99.4% in all the test, I presume due to saturation. On the other hand, the results on CIFAR10 can be summed up as follows:
  * Baseline            : 65.82%
  * Weight Decay        : 68.06%
  * Dropout             : 68.12%
  * ImageNorm           : 69.29%
  * BatchNorm           : 73.66%
  * Data Augmentation   : 74.30% (Reszied to 32x32 with padding 4 + RandomHorizontalFlip with p = 0.5)
  * Dropout + ImageNorm : 74.46%
  * The results for all the above can be found [here](https://github.com/DevChuriwala/FuzzyAdventure/tree/master/LeNet%20CV/Benchmarking%20Results).
  
### Bonus Data Augmentation Task (iii):
For testing Data Augmentation on MNIST I used the RandomHorizontalFlip (with a probability of 0.5) transform to augment the dataset. Using that, the model achieved an accuracy of **99.01%**, slightly lower than when I used no flips. I attribute this decrease to the fact that, the data becomes noisy for no good reason. Flipping might aid in detecting images of numbers such as 0,1,8 but in all other cases it adds noise to well strucured data that is being used to train. A better augmentation technique in my opinion, for MNIST, is RandomRotation(smalltheta), because numbers are often written at slight angles and adding a small rotation to the training data might help with increasing the accuracy.
  
  
  
