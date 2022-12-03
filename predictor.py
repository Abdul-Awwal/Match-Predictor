
# import required modules
import pandas as pd
import numpy as np
import time
from dask import dataframe as df1

def main(home_team, away_team, week):
    # time taken to read data
    s_time_dask = time.time()
    dask_df = df1.read_csv('eplmatches.csv')
    e_time_dask = time.time()
    
    print("Read with dask: ", (e_time_dask-s_time_dask), "seconds")
    print(home_team, away_team, week)
    
    # data
    dask_df.head(10)
    

if __name__ == "__main__":
    main("Liverpool", "Chelsea", 4)