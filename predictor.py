
# import required modules
import pandas as pd
import numpy as np
import time
from dask import dataframe as df1

def main_best_algorithm(home_team, away_team, week):
    
    #Get data
    # time taken to read data
    s_time_dask = time.time()
    dask_df = df1.read_csv('eplmatches.csv')
    e_time_dask = time.time()
    
    print("Read with dask: ", (e_time_dask-s_time_dask), "seconds")
    print(home_team, away_team, week)
    
    # data
    dask_df.head(10)
    
    #Preprocess Data

    data = pd.read_csv('eplmatches.csv')
    print(data)
    
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
    
    
    
    #Call Naive Bayes and print results
    
    #Run Decision Tree Algorithem and print results
    
    
    #Compare scores of Naive bayes and Decision tree
    #if Naive Bayes better
        #Print Naive Bayes
    #else Decsion tree better
        #print decison tree
    

if __name__ == "__main__":
    main_best_algorithm("Liverpool", "Chelsea", 4)
    