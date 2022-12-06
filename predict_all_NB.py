# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:48:05 2022

@author: Sarah
"""


# import required modules
import pandas as pd
import numpy as np
import time
from dask import dataframe as df1
from sklearn import preprocessing, tree, naive_bayes, metrics
def main_best_algorithm():
    #get user input
    #week = input('Input match week number: ')
    #home_team = input('Input home team: ')
    #away_team = input('Input away team: ')
    
    
    #Get data
    # time taken to read data
    s_time_dask = time.time()
    dask_df = df1.read_csv('eplmatches.csv')
    e_time_dask = time.time()
    le1 = preprocessing.LabelEncoder() # for home and away colums, for test and training
    le2 = preprocessing.LabelEncoder() # for ftr column, for test and training

    print("Read with dask: ", (e_time_dask-s_time_dask), "seconds")
    #print(home_team, away_team, week)
    #print(type(week))
    
    # data
    dask_df.head(10)
    
    #Preprocess Data
    #write input to acsv file
   # match_input ={'Wk': [week],'Home': [home_team],'Away':[away_team],'FTR':[ftr]}
    
    # creating dataframe from the above dictionary of lists
    #dataFrame = pd.DataFrame(match_input)
    #print("DataFrame...",dataFrame)
    #dataFrame.to_csv("test_matches.csv")

    data = pd.read_csv('eplmatches.csv')
    train = pd.read_csv('eplmatches.csv')
    test = pd.read_csv('2022_eplmatches.csv')


    
    #print(data)
    
    #drop columns
    train.drop('Season_End_Year', inplace=True, axis=1)

    #date
    train.drop('Date', inplace=True, axis=1)

    #home goals
    train.drop('HomeGoals', inplace=True, axis=1)

    #away goals
    train.drop('AwayGoals', inplace=True, axis=1)

    # Match Number 
    test.drop('Match Number', inplace=True, axis=1)
    test.drop('Date', inplace=True, axis=1)
    test.drop('Location', inplace=True, axis=1)
    test.drop('Result', inplace=True, axis=1)

    
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
    #le1.fit()
    le2.fit(data.get('FTR'))

    #transform train and test
    train.insert(1,'Home',le1.transform(train.pop('Home')))
    
    train.insert(2,'Away',le1.transform(train.pop('Away')))
    
    train.insert(3, 'FTR',le2.transform(train.pop('FTR')))
    
    test.insert(1,'Home',le1.transform(test.pop('Home')))
    
    test.insert(2,'Away',le1.transform(test.pop('Away')))
    
    #test.insert(3, 'FTR',le2.transform(test.pop('FTR')))


    print(train)
    print(test)
    
    #Call Naive Bayes and print results
    test_predictions = naiveBayes(train, test,le2)
    
    test.insert(1,'Home',le1.inverse_transform(test.pop('Home')))
    
    test.insert(2,'Away',le1.inverse_transform(test.pop('Away')))
    print(test)
    write_predictions(test_predictions, test)
    
    
def naiveBayes(dataset_train, dataset_test,le2):
    print('in naive bayes')
    #seperate FTR out to variable for train
    ftr_train = dataset_train.pop('FTR')
    #print(ftr_train)
    #print(dataset_train)
    
    #seperate FTR out to variable for test
    #ftr_test = dataset_test.pop('FTR')
    #print(ftr_test)
    #print(dataset_test)
    
    #create model using train
    categorical = naive_bayes.CategoricalNB() 
    categorical.fit(dataset_train, ftr_train)
    
    #use model to predict test
    predictions = categorical.predict(dataset_test)
    print("predictions")
    
    predictions2 = le2.inverse_transform(predictions)
    print(predictions2)
    
    #accuracy = metrics.accuracy_score(ftr_test, predictions)
    #print(accuracy)
    #f1 = metrics.f1_score(ftr_test, predictions, average=None)
    #print(f1) # one value for draw, away, home each
    #precision = metrics.precision_score(ftr_test, predictions, average = None,zero_division=0)
    #print(precision)
    
    dataset_train.insert(3, 'FTR',ftr_train)
    #dataset_test.insert(3, 'FTR',ftr_test)
    return predictions2

def write_predictions(predictions, dataset):
    #write input to acsv file
    
    #weeks
    weeks= dataset.get('Wk')
    week_values = weeks.values
    week_list = week_values.tolist()
    #print(week_list)
    
    #home
    home= dataset.get('Home')
    home_values = home.values
    home_list = home_values.tolist()
    #print(home_list)
    
    #away
    away= dataset.get('Away')
    away_values = away.values
    away_list = away_values.tolist()
    #print(away_list)
    
    match_input ={'Wk': week_list,'Home': home_list,'Away': away_list,'Predictions':predictions}

    # creating dataframe from the above dictionary of lists
    dataFrame = pd.DataFrame(match_input)
    print("DataFrame...",dataFrame)
    dataFrame.to_csv("2022_match_predictions.csv")
    

if __name__ == "__main__":
    main_best_algorithm()
    