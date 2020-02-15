![Build Status](https://travis-ci.org/jonthemango/big-data.svg?branch=master)
[![Code Coverage](https://codecov.io/github/jonthemango/big-data/coverage.svg)](https://codecov.io/gh/jonthemango/big-data)

## Authors
Jon Mongeau - https://github.com/jonthemango

Ribal Aladeeb - https://github.com/ribal-aladeeb

Feras Younis - https://github.com/FerasYounis


# Abstract
Many people who lack an extensive credit history find it exceedingly difficult to receive a loan. As a result, these people get taken advantage of by shady lenders. The purpose of this project is to use alternative data about customers to determine whether or not they have the ability to repay their loans. This is a supervised binary classification problem. 

The [dataset](https://www.kaggle.com/c/home-credit-default-risk/overview) is provided by Kaggle for a competition. The training data contains 300K+ records of historical customer data labeled 0 or 1 for having repaid the loan or not. There are 122 available columns available to select our features from. The point of the project is to use this historical data to accurately predict if customers can repay loans without requiring their credit history.


# Introduction

## Context
[Home Credit](http://www.homecredit.net/about-us.aspx) aims to offer loaning services to unbanked and underbanked people. Their target clientele could be either new immigrants, people of lower income, people who recently got out of homelessness or anyone who simply has not had access to banking services. The company posted this dataset of historical loans that they have given out along with all the information that they collect regarding those loans in the context of a kaggle competition. The competition is now expired but you can still see the top submissions and there is some good documentation on where to get started.

## Objectives
The main goal is to find the best predictors to determine whether a client can repay a loan, given that they do not have a substantial credit history. The company provides an extensive amount of data regarding their customers across multiple tables. Our job is to find a winning combination of features across all tables that will yield the best precision and recall possible. The winning score on kaggle is 0.80570 using the ROC curve.


## Presentation of the Problem to Solve
This project is a supervised binary classification learning problem. Our first problem is that a lot of columns are missing values, some are very sparse. For this we wrote a quick [script](https://github.com/jonthemango/big-data/blob/master/preprocessing/step1_column_analysis.py) that counts missing values by column for a table and generates a [json](https://github.com/jonthemango/big-data/blob/master/preprocessing/missing_values.json) and [csv](https://github.com/jonthemango/big-data/blob/master/preprocessing/missing_values.csv) report. Our plan for now is to identify all columns with missing values and drop them. We feel like we can afford to do this because our main application table contains 122 columns and 55 out of them do not contain missing values.

Furthermore, we have a combination of about 50M+ records across 7 tables. The common denominator is a column named SK_ID_CURR which denotes the ID of a current loan application. This ID is unique in the application_train.csv file but can be repeated in other tables, which creates multiplicity. A big challenge is to aggregate the data related to a given SK_ID_CURR from all tables into a single feature vector for training.

## Related Work
- [Introduction to Dataset/Problem](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction)

- [Feature Engineering ](https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics)


# Materials and Methods

## The Dataset
The entire dataset emcompasses 7 csv files.  

| File Name  | Shape | Description | Imfortant Features
|---|---|---|---|
| **./data/application_train.csv**  | 307,511 rows, 122 columns  | Contains information about the approval of loan applications. Each row is a loan application. | `SK_ID_CURR` identifies the loan application. `TARGET` is a 0 if the loan was repaid while a 1 determines whether the loan is not repaid. |
| ./data/bureau.csv | 1,716,428 rows, 17 columns | Contains information about loan applicants' credit history from other banks and financial institutions. Each row is a credit.  | `SK_ID_BUREAU` identifies the credit. `SK_ID_CURR` identifies the loan application. |
| ./data/bureau_balance.csv | 27,299,925 rows, 3 columns  | Contains monthly data about credits from `bureau.csv`. Each row is a monthly credit. | `SK_ID_BUREAU` identifies the credit. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent) |
| ./data/previous_application.csv | 1,670,214 rows, 37 columns | Each row represents one previous loan application. There may be many previous loan applications for every current application in `application_train.csv`  | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application.  |
| ./data/POS_CASH_BALANCE.csv | 10,001,358 rows, 8 columns | Contains monthly data about previous POS and cash loans. Each row is a month of a loan. | `SK_ID_PREV` identifies the previous loan, `SK_ID_CURR` identifies the current application loan. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent) |
| ./data/credit_card_balance.csv | 3,840,312 rows, 23 columns | Contains monthly data about credit cards clients had with the company. Each row is a monthly credit card balance. | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent). |
| ./data/installments_payments.csv | 13,605,401 rows, 8 columns | Contains payment history data for previous loans applications. Each row is a payment made or missed.  | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application. `AMT_PAYMENT` identifies the amount payed (0.0 means missed). |

Additionally we are provided with `HomeCredit_columns_descriptions.csv` which provides an in-depth description for the column names in each table.

An interesting note about the available data is the distribution between loans which have been repaid and loans which have not been repaid.
```
>>> df = data.select("TARGET")
>>> not_repaid = df.where(data["TARGET"] == 1).count() # 1 means client did not repay loan
>>> repaid = df.where(data["TARGET"] == 0).count() # 0 means loan was repaid
# 282,686 repaid, 24,825 not_repaid
>>> repaid/(repaid+not_repaid)
0.9192711805431351
```
Meaning 91.93 % loans get repaid. We will need to take this into consideration when classifying our data. One class of loan applications is far larger than the other. We will either need to sub-sample our data (after already doing so after seperating into train/test) or we will need to add linear combinations of approved loans. We will experiment with both approaches to see what makes our model most effective.

## Technologies and Algorithms
We will use Apache Spark Dataframes and RDDs in order to perform some pre-processing on the dataset. For our feature engineering our goal is to perform dynamic feature engineering. The goal with this is to evaluate which features yield the best model. We will seperate the data into 2/3 train and 1/3 test data. We want to also attempt to do this dynamically. Feature engineering can involve feature construction and feature selection, there is one technique we would to try for construction: Polynomial features, In this method, we make features that are powers of existing features as well as interaction terms between existing features. By combining two variables together into a single interaction may show stronger relation with the target “client may or not repay the loan”. Then, to get a baseline, and after encoding the categorical variables. Will use all features and fill the missing data and normalizing the range of feature. Scikit tool can be used to preprocessing these two steps on the data.

For training we will use Skikit-Learn. We are interested in applying packaged algorithms like Descision Trees (Random Forests), kNN and SVMs. We will experiment with other algorithms as we see fit but our plan is to commence with these algorithms. We also plan on using a random classifier in order to establish a baseline.

For evaluation we will define true positives as loans we correctly predict as accepted, true negatives as loans we correctly predict as rejected, false positives as loans we incorrectly predict as accepted and false negatives as loans we incorrectly predict as rejected. We will use Skikit-Learn to evaluate our models.

Once we iterate upon this process of pre-process to test/train split to algorithm-selections to evaluation we will also be deploying our model to a web service continuously. The goal is to wrap a Python Flask web service around a sub-set of our models and create a Web UI where users can enter information of a client and view what values our models predict. We will use docker, Flask and ReactJs for this web service.
