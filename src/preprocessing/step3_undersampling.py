import sys
sys.path.append(".")

from src import utils
import json
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.rdd import RDD



root_dir = utils.get_project_root_dir()
spark = utils.init_spark()


def undersample(train_df, class_ratio=0.8):
    '''
    This will undersample the training set with respect to the majority to
    minority class_ratio mentionned above. The original set has a ratio of 0.92.
    Keep in mind that class_ratio is not the portio of the majority class that
    will be dropped. It is the desired outcome ratio. This function will take
    care of figuring out the exact sampling_ratio necessary to achieve the
    desire class_ratio. You can then experiement with different class ratios to
    see what yields the best results.
    '''
    original_ratio = 0.92  # statistically this will hold true
    total_count = train_df.count()
    majority_class_df = train_df.where(train_df['TARGET'] == 0)
    minority_class_df = train_df.where(train_df['TARGET'] == 1)
    majority_count = majority_class_df.count()
    print(f'\n\nminority_class_df.count() = {minority_class_df.count()}')
    print(f'number of majority class records =  {majority_count}')

    minority_count = total_count-majority_count

    records_to_be_kept = class_ratio*minority_class_df.count()/(1-class_ratio)
    sampling_ratio = records_to_be_kept/majority_count

    undersampled_majority = majority_class_df.sample(sampling_ratio)
    new_train_df = undersampled_majority.union(minority_class_df)
    return new_train_df


if __name__ == '__main__':

    df = spark.read.csv(f'{root_dir}data/application_train.csv', header=True)
    df.cache()
    undersampled_df = undersample(df, class_ratio=0.8)
    undersampled_df.cache()

    original_count = df.count()
    us_count = undersampled_df.count()

    print('\n\n')
    print(f'original count = {original_count}')
    print(f'undersampled count = {us_count}')

    maj = df.where(df['TARGET'] == 0).count()
    total = undersampled_df.count()
    new_maj = undersampled_df.where(undersampled_df['TARGET'] == 0).count()
    new_minority = undersampled_df.where(
        undersampled_df['TARGET'] == 1).count()
    assert(total == new_maj+new_minority)
    ratio = round(new_maj/(new_maj+new_minority)*100, 4)

    print(f'\nminority count after sampling = {new_minority}')
    print(f'minority count before sample = {df.where(df["TARGET"]==1).count()}  == {new_minority}')

    print(f'old majority count {maj}')
    print(f'new majority count {new_maj}')
    print(f'ratio = {ratio}%')
