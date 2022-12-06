import pandas as pd
import numpy as np
import time
from dask import dataframe as df1
from sklearn import preprocessing
def main_best_algorithm(home_team, away_team, week):
    
    #Get data
    # time taken to read data
    s_time_dask = time.time()
    dask_df = df1.read_csv('train.csv')
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
    
    
    #fit with data
    le1.fit(data.get('Home'))
    le2.fit(data.get('FTR'))
    data.insert(1,'Home',le1.transform(data.pop('Home')))
    data.insert(3,'FTR',le1.transform(data.pop('FTR')))
    name = le1.inversetransform(0)
    winloss(name, le1, data)

def winloss(name, data):
    number = le1.transform(name)
    loss = 0
    win = 0
    draw = 0
    for index, row in data.head().iterrow():
        if row['Home'] == number:
            if row['FTR'] == 2:
                win += 1
            elif row['FTR'] == 0:
                loss += 1
            else:
                draw += 1
    
    print("win, loss, draw")
    print(win)
    print(loss)
    print(draw)

if __name__ == "__main__":
    main_best_algorithm("Liverpool", "Chelsea", 4)
