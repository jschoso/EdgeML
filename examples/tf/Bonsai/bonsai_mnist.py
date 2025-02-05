# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import helpermethods
import tensorflow as tf
import numpy as np
import sys
import os

#Provide the GPU number to be used
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

#Bonsai imports
from edgeml_tf.trainer.bonsaiTrainer import BonsaiTrainer
from edgeml_tf.graph.bonsai import Bonsai

# Fixing seeds for reproducibility
tf.set_random_seed(42)
np.random.seed(42)

#Loading and Pre-processing dataset for Bonsai
dataDir = "/home/jschosto/EdgeML/examples/tf/Bonsai/MNIST-10"
(dataDimension, numClasses, Xtrain, Ytrain, Xtest, Ytest, mean, std) = helpermethods.preProcessData(dataDir, isRegression=False)
print("Feature Dimension: ", dataDimension)
print("Num classes: ", numClasses)

sigma = 1.0 #Sigmoid parameter for tanh
depth = 3 #Depth of Bonsai Tree
projectionDimension = 28 #Lower Dimensional space for Bonsai to work on

#Regularizers for Bonsai Parameters
regZ = 0.0001
regW = 0.001
regV = 0.001
regT = 0.001

totalEpochs = 100

learningRate = 0.01

outFile = None

#Sparsity for Bonsai Parameters. x => 100*x % are non-zeros
sparZ = 0.2
sparW = 0.3
sparV = 0.3
sparT = 0.62

batchSize = np.maximum(100, int(np.ceil(np.sqrt(Ytrain.shape[0]))))

useMCHLoss = True #only for Multiclass cases True: Multiclass-Hing Loss, False: Cross Entropy. 

#Bonsai uses one classier for Binary, thus this condition
if numClasses == 2:
    numClasses = 1



X = tf.placeholder("float32", [None, dataDimension])
Y = tf.placeholder("float32", [None, numClasses])

currDir = helpermethods.createTimeStampDir(dataDir)
helpermethods.dumpCommand(sys.argv, currDir)

bonsaiObj = Bonsai(numClasses, dataDimension, projectionDimension, depth, sigma)

bonsaiTrainer = BonsaiTrainer(bonsaiObj, regW, regT, regV, regZ, sparW, sparT, sparV, sparZ,
                              learningRate, X, Y, useMCHLoss, outFile)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())#

bonsaiTrainer.train(batchSize, totalEpochs, sess,
                    Xtrain, Xtest, Ytrain, Ytest, dataDir, currDir)



