
# For RQ1, determines whether Naive Bayes or Decision Tree is better

# import required modules
import pandas as pd
import numpy as np
import time
from dask import dataframe as df1
from sklearn import preprocessing, tree, naive_bayes, metrics
def main_best_algorithm(home_team, away_team, week):
    
    #Get data
    # time taken to read data
    s_time_dask = time.time()
    dask_df = df1.read_csv('eplmatches.csv')
    e_time_dask = time.time()
    le1 = preprocessing.LabelEncoder() # for home and away colums, for test and training
    le2 = preprocessing.LabelEncoder() # for ftr column, for test and training

    print("Read with dask: ", (e_time_dask-s_time_dask), "seconds")
    print(home_team, away_team, week)
    
    # data
    dask_df.head(10)
    
    #Preprocess Data

    data = pd.read_csv('eplmatches.csv')
    matches = pd.read_csv('eplmatches.csv')
    print('matches:')
    print(matches)
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    #print(data)
    
    #drop columns
    train.drop('Season_End_Year', inplace=True, axis=1)
    test.drop('Season_End_Year', inplace=True, axis=1)

    #date
    train.drop('Date', inplace=True, axis=1)
    test.drop('Date', inplace=True, axis=1)

      
    #home goals
    train.drop('HomeGoals', inplace=True, axis=1)
    test.drop('HomeGoals', inplace=True, axis=1)


    #away goals
    train.drop('AwayGoals', inplace=True, axis=1)
    test.drop('AwayGoals', inplace=True, axis=1)

    
    print("\nCSV Data after deleting the columns:\n")
    print(train)
    print(test)
    
    #convert teams to numbers
    #dataset with words
    #pop column hometeam
    #le1.fit(data.pop('Home'))
    #print("home popped")
    #print(data.pop('Home'))
    #print(data)
    
    #labelencoder variable with home team
    #insert transformed clumn back into data
    #repeat awayteam and ftr
    
    #fit with data
    le1.fit(data.get('Home'))
    le2.fit(data.get('FTR'))

    #transform train and test
    train.insert(1,'Home',le1.transform(train.pop('Home')))
    
    train.insert(2,'Away',le1.transform(train.pop('Away')))
    
    train.insert(3, 'FTR',le2.transform(train.pop('FTR')))
    
    test.insert(1,'Home',le1.transform(test.pop('Home')))
    
    test.insert(2,'Away',le1.transform(test.pop('Away')))
    
    test.insert(3, 'FTR',le2.transform(test.pop('FTR')))


    print(train)
    print(test)
    
    #Call Naive Bayes and print results
    naiveBayes(train, test)
    
    #Run Decision Tree Algorithem and print results
    decisionTree(train, test)
    
    #Compare scores of Naive bayes and Decision tree
    #if Naive Bayes better
        #Print Naive Bayes
    #else Decsion tree better
        #print decison tree
    
def naiveBayes(dataset_train, dataset_test):
    print('in naive bayes')
    #seperate FTR out to variable for train
    ftr_train = dataset_train.pop('FTR')
    #print(ftr_train)
    #print(dataset_train)
    
    #seperate FTR out to variable for test
    ftr_test = dataset_test.pop('FTR')
    #print(ftr_test)
    #print(dataset_test)
    
    #create model using train
    categorical = naive_bayes.CategoricalNB() 
    categorical.fit(dataset_train, ftr_train)
    
    #use model to predict test
    predictions = categorical.predict(dataset_test)
    accuracy = metrics.accuracy_score(ftr_test, predictions)
    print(accuracy)
    f1 = metrics.f1_score(ftr_test, predictions, average=None)
    print(f1) # one value for draw, away, home each
    precision = metrics.precision_score(ftr_test, predictions, average = None)
    print(precision)
    matrix = metrics.confusion_matrix(ftr_test, predictions)
    print(matrix)
    dataset_train.insert(3, 'FTR',ftr_train)
    dataset_test.insert(3, 'FTR',ftr_test)



def decisionTree(dataset_train, dataset_test):
    print('in decsion tree')
    #seperate FTR out to variable for train
    ftr_train = dataset_train.pop('FTR')
    #print(ftr_train)
    #print(dataset_train)
    
    #seperate FTR out to variable for test
    ftr_test = dataset_test.pop('FTR')
    #print(ftr_test)
    #print(dataset_test)
    
    #create model using train
    decision = tree.DecisionTreeClassifier() 
    decision.fit(dataset_train, ftr_train)
    
    #use model to predict test
    predictions = decision.predict(dataset_test)
    accuracy = metrics.accuracy_score(ftr_test, predictions)
    print()
    print(accuracy)
    f1 = metrics.f1_score(ftr_test, predictions, average=None)
    print(f1) # one value for draw, away, home each
    precision = metrics.precision_score(ftr_test, predictions, average = None)
    print(precision)
    matrix = metrics.confusion_matrix(ftr_test, predictions)
    print(matrix)


if __name__ == "__main__":
    main_best_algorithm("Liverpool", "Chelsea", 4)
    