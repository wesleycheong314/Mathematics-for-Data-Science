"""
Tools for loading the MNIST Data.

@author: Brett
"""

import numpy as np
from mnist_tools import *
from plot_tools import *

"""
Given train (in the format returned by load_train_data in mnist_tools), 
and a 1d numpy array testImage you should return a tuple (digit,imageIdx).  digit is
an integer giving the numerical digit value of the training image closest 
to the testImage in Euclidean distance.  imageIdx is the row number of the closest 
training image in the 2d array train[digit].
"""
def compute_nearest_neighbors(train, testImage) :
    #Your code here
    return None

"""
Assumes the data file is in 'mnist_all.mat'.
"""
def main() :
    datafile = "mnist_all.mat" #Change if you put the file in a different path
    train = load_train_data(datafile)
    test,testLabels = load_test_data(datafile)
    imgs = []
    estLabels = []
    for i in range(len(testLabels)) :
        trueDigit = testLabels[i]
        testImage = test[i,:]
        (nnDig,nnIdx) = compute_nearest_neighbors(train,testImage)
        imgs.extend( [testImage,train[nnDig][nnIdx,:]] )
        estLabels.append(nnDig)

    row_titles = ['Test','Nearest']
    col_titles = ['%d vs. %d'%(i,j) for i,j in zip(testLabels,estLabels)]
    plot_image_grid(imgs,
                    "Image-NearestNeighbor",
                    (28,28),len(testLabels),2,True,row_titles=row_titles,col_titles=col_titles)

if __name__ == "__main__" :
    main()
