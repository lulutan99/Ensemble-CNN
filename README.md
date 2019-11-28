# Ensemble-CNN
Ensemble convolution neural network to classify images:

* The data was pre-processed using a heuristic segmentation approach that identified the largest bounding box.
Firstly, several filters were applied and the image was converted to binary. Once the largest binary digit was identified, it was extracted from the original (i.e. unprocessed) image and scaled to size (i.e. 64x64 pixels.) This was then fed into the network as the training data. 

* A stack of 3 convolution neural network is ensembled together to improve classification accuracy