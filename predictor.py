
# import required modules
import pandas as pd
import numpy as np
import time
from dask import dataframe as df1
from sklearn import preprocessing, tree, naive_bayes
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
    #print(data)
    
    #drop columns
    data.drop('Season_End_Year', inplace=True, axis=1)
    #date
    data.drop('Date', inplace=True, axis=1)
      
    #home goals
    data.drop('HomeGoals', inplace=True, axis=1)

    #away goals
    data.drop('AwayGoals', inplace=True, axis=1)

    
    print("\nCSV Data after deleting the column 'year':\n")
    print(data)
    
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
    
    le1.fit(data.get('Home'))
    
    data.insert(1,'Home',le1.transform(data.pop('Home')))
    
    data.insert(2,'Away',le1.transform(data.pop('Away')))
    
    le2.fit(data.get('FTR'))
    
    data.insert(3, 'FTR',le2.transform(data.pop('FTR')))

    print(data)
    
    #Call Naive Bayes and print results
    NaiveBayes(data)
    #Run Decision Tree Algorithem and print results
    
    
    #Compare scores of Naive bayes and Decision tree
    #if Naive Bayes better
        #Print Naive Bayes
    #else Decsion tree better
        #print decison tree
    
def NaiveBayes(dataset):
    print('in naive bayes')
    #seperate FTR out to variable
    ftr = dataset.pop('FTR')
    print(ftr)
    print(dataset)
    
    categorical = naive_bayes.CategoricalNB() 
    categorical.fit(dataset, ftr)
    print(categorical)
    #predictions = categorical.pred

if __name__ == "__main__":
    main_best_algorithm("Liverpool", "Chelsea", 4)
    