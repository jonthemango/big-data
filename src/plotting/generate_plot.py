import sys
sys.path.append(".")

import os
from operator import itemgetter
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from src import utils

files = ['rf_predictions.csv']
root_dir = utils.get_project_root_dir()
spark = utils.init_spark()

def df_from_disk(filename):
    # prefix = 'file://' if sys.platform == 'linux' else ''
    prefix=''
    from_disk_df = spark.read.csv(f'{prefix}{root_dir}/results/{filename}',header=True)
    return from_disk_df

def write_actives():
    df = df_from_disk(files[0])
    actives = df.where(df['TARGET']==1).select('SK_ID_CURR').collect()
    print(actives[:10])

write_actives()

