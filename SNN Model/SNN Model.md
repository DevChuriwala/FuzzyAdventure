# SNN Model

## i) Framing a Question

As time progresses our Neural Networks are becoming deeper and there is tons of data to process. Conventionally, computing this via ANNs consumes a lot of energy, which is often not possible on edge//IOT devices. Hence, we seek to find a way of creating an energy-efficient and accurate model to classify images. The ideal starting point to base our model off of is the human brain which consumes only 12 watts - minuscule compared to the GPUs we use to train ANNs. So, two questions arise,(1) Can we mimic the brain's efficiency in our model whilst maintaining the accuracy of conventional models? And (2) How can we implement the best ideas from previously published work on SNN to optimize our model and reduce the existing gap between ANNs and SNNs? To answer both the questions, we first need to understand the current State of the art (SOTA) in the field of SNNs.

## ii) Understanding the SOTA

