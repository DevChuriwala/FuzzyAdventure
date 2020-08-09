# SNN Model

## i) Framing a Question

As time progresses our Neural Networks are becoming deeper and there is tons of data to process. Conventionally, computing this via ANNs consumes a lot of energy, which is often not possible on edge//IOT devices. Hence, we seek to find a way of creating an energy-efficient and accurate model to classify images. The ideal starting point to base our model off of is the human brain which consumes only 12 watts - minuscule compared to the GPUs we use to train ANNs. So, two questions arise,(1) Can we mimic the brain's efficiency in our model whilst maintaining the accuracy of conventional models? And (2) How can we implement the best ideas from previously published work on SNN to optimize our model and reduce the existing gap between ANNs and SNNs? To answer both the questions, we first need to understand the current State of the art (SOTA) in the field of SNNs.

## ii) Understanding the SOTA

<img src="./Images/SNN_Class.png" width="800">
As shown above, the SOTA models for SNNs currently consist of (a)Converted, (b)Unsupervised and (c)Supervised SNNs.<br>
(a) Converted SNNs use advanced ANN optimization techniques and boast nearly ANN level accuracies(Rueckauer et al., 2017; Sengupta et al., 2019) but the problem arises in the signal representation, wherein the frequencies need to be estimated requiring non-trivial passage on time,<br>
(b) Unsupervised methods such as STDP are biologically plausible and show high accuracies on datasets such as MNIST(Diehl and Cook, 2015) but the method does not scale well to deeper networks and hence has limited expressivity,<br>
(c) Supervised methods of training on Spike-Based Data requires non-conventional BP, but it shows promise in terms of energy efficiency and better generalization, thus acheiving better accuracies.<br>
