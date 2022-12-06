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

    data = pd.read_csv('train.csv')
    #print(data)
    
    
    #fit with data
    le1.fit(data.get('Home'))
    le2.fit(data.get('FTR'))
    data.insert(3,'Home',le1.transform(data.pop('Home')))
    data.insert(6,'Away',le1.transform(data.pop('Away')))
    data.insert(7,'FTR',le2.transform(data.pop('FTR')))
    teamone= input('Input first team: ')
    teamtwo= input('Input second team: ')
    name = [teamone]
    name1= [teamtwo]
    winloss(name, name1, le1, data)

def winloss(name, name1, le1, data):
    number = le1.transform(name)
    number1 = le1.transform(name1)
    loss = 0
    win = 0
    draw = 0

    for index, row in data.iterrows():
        if row['Home'] == number and row['Away']==number1:
            if row['FTR'] == 2:
                win += 1
            elif row['FTR'] == 0:
                loss += 1
            else:
                draw += 1
        if row['Away'] == number and row['Home']==number1:
            if row['FTR'] == 2:
                loss += 1
            elif row['FTR'] == 0:
                win += 1
            else:
                draw += 1
    print(le1.inverse_transform(number))
    print("win, loss, draw")
    winratio=0
    drawratio=0
    lossratio=0
    winratio= win/(win+loss+draw)
    lossratio= loss/(win+loss+draw)
    drawratio=draw/(win+loss+draw)
    approx=round(winratio,4)
    approxdraw=round(drawratio,4)
    approxloss=round(lossratio,4)

    print(approxdraw)
    print(approx)
    print(approxloss)
    print(draw)
    print(name,' has a ', approx *100, '% chance of winning, ', 'a ', approxloss*100, '% chance of losing, and a ',
          round(drawratio*100,2), '% chance of drawing.')

if __name__ == "__main__":
    main_best_algorithm("Liverpool", "Chelsea", 4)
