import numpy
import random
import time
import pandas as pd
from math import sqrt, pi, exp

## You may have to select/assign a path for training and testing file
trainingFile = "/irisTraining.txt"
testingFile = "/irisTesting.txt"
Xtrain = numpy.loadtxt(trainingFile)
n = Xtrain.shape[0]
d = Xtrain.shape[1]-1
print (n, d)

#Training... Collect mean and standard deviation for each dimension for each class..
#Also, calculate P(C+) and P(C-)

X = Xtrain[:, 0:-1]
X = pd.DataFrame(X, columns = list(range(0, d)))
y = Xtrain[:, -1]

# mean of a list of numbers
def mean(numList):
  return sum(numList)/float(len(numList))

# standard deviation of a list of numbers
def stdev(numList):
  average = mean(numList)
  variance = sum([(x-average)**2 for x in numList]) / float(len(numList)-1)
  standard_deviation = sqrt(variance)
  return standard_deviation

# Gaussian PDF for x = (1/sqrt(2*pi)*(stdev))*exp(-((x-mean)^2/(2*stdev^2)))
def gaussianPDF(x, mean, stdev):
	return 1/(sqrt(2*pi)*stdev)*exp(-((x-mean)**2/(2*stdev**2)))

positiveClass = X[y == 1]
negativeClass = X[y==-1]

# Collecting mean and standard deviation for each dimension for each class..

positiveClassStatistics = [(mean(positiveClass[column]),
                            stdev(positiveClass[column])) for column in positiveClass]

negativeClassStatistics = [(mean(negativeClass[column]),
                            stdev(negativeClass[column])) for column in negativeClass]

# Calculate the probabilities of predicting each class for a given row
def calculateProbabilities(row):
  total_rows = X.shape[0]
  probabilityPositiveClass = positiveClass.shape[0]/total_rows
  for i in range(positiveClass.shape[1]):
    probabilityPositiveClass *= gaussianPDF(row[i],
                                            positiveClassStatistics[i][0],
                                            positiveClassStatistics[i][1])
  probabilityNegativeClass = negativeClass.shape[0]/total_rows
  for i in range(negativeClass.shape[1]):
    probabilityNegativeClass *= gaussianPDF(row[i],
                                            negativeClassStatistics[i][0],
                                            negativeClassStatistics[i][1])
    
  return probabilityPositiveClass, probabilityNegativeClass

#Testing .....
Xtest = numpy.loadtxt(testingFile)
nn = Xtest.shape[0] # Number of points in the testing data.
dd = Xtest.shape[1]-1

y_actual = Xtest[:, -1]
Xtest = Xtest[:, 0:-1]

y_pred = [1 if calculateProbabilities(row)[0] > calculateProbabilities(row)[1] else -1 for row in Xtest]

for i in range(len(Xtest)):
  print('Xi: {}\nP(C+|Xi) => P(Xi|C+)*P(C+) = {}, P(C-|Xi) => P(Xi|C-)*P(C-) = {}'.format(Xtest[i],
                                                                                          calculateProbabilities(Xtest[i])[0],
                                                                                          calculateProbabilities(Xtest[i])[1]))
  print('Prediction = {}, Actual = {}'.format(y_pred[i], y_actual[i]))
  print()


tp = sum([1 if y_pred[i] == 1 and y_actual[i] == 1 else 0 for i in range(len(y_pred))]) #True Positive
fp = sum([1 if y_pred[i] == 1 and y_actual[i] == -1 else 0 for i in range(len(y_pred))]) #False Positive
tn = sum([1 if y_pred[i] == -1 and y_actual[i] == -1 else 0 for i in range(len(y_pred))]) #True Negative
fn = sum([1 if y_pred[i] == -1 and y_actual[i] == 1 else 0 for i in range(len(y_pred))]) #False Negative

print('True Positive: {}\nFalse Positive: {}\nTrue Negative: {}\nFalse Negative: {}'.format(tp, fp, tn, fn))


precision = tp/(tp+fp)
recall = tp/(tp+fn)

print('Precision: {}\nRecall: {}'.format(precision, recall))

#Iterate over all points in testing data
#For each point find the P(C+|Xi) and P(C-|Xi) and decide if the point belongs to C+ or C-..
#Recall we need to calculate P(Xi|C+)*P(C+) ..
#P(Xi|C+) = P(Xi1|C+) * P(Xi2|C+)....P(Xid|C+)....Do the same for P(Xi|C-)
#Now that you've calculate P(Xi|C+) and P(Xi|C-), we can decide which is higher
#P(Xi|C-)*P(C-) or P(Xi|C-)*P(C-) ..
#increment TP,FP,FN,TN accordingly, remember the true lable for the ith point is in Xtest[i,(d+1)]

#}

#Calculate all the measures required..