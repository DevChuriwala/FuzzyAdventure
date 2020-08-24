# NLP - Reddit Flair Detection


The task required me to design a RNN using LSTM. I created one as follows:
  * I first used the pandas library to parse the data.
  * A key observation made during data exploration was that, the dataset was extremely skewed with Class 0 containing nearly twice the number of samples as its closest rival, Class 7, which itself contained roughly 1.5x the number of samples of the nearest rival, Class 10.
  * Then I went on to combine the 'Title' and 'Post' columns into a new 'Combine column.'
  * I used Spacy to tokenize the words, removing all the punctuations/numbers etc.
  * I then created a Vocab2index dict to encode my samples, to vectors.
  * The model architecture consisted of 2 LSTM layers with hyperparameters defined as: Batchsize = 5000; Epochs = 30; LR = 0.3, Embedding_Dim = 50 and Hdden_Dim = 50.
  * I used the SKlearn metrics to calculate the final F1 scores, which came out to MicroF1 = 0.46 and Macro F1 = 0.072 (the model was extremely under-optimized).
  * The notebook for the above can be found [here](https://github.com/DevChuriwala/FuzzyAdventure/blob/master/NLP%20Reddit%20Flair%20Detector/NLP_Reddit.ipynb).
