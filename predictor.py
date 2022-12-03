
# import required modules
import pandas as pd
import numpy as np
import time
from dask import dataframe as df1

# time taken to read data
s_time_dask = time.time()
dask_df = df1.read_csv('dataset.csv')
e_time_dask = time.time()

print("Read with dask: ", (e_time_dask-s_time_dask), "seconds")

# data
dask_df.head(10)