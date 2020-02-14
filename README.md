![Build Status](https://travis-ci.org/jonthemango/big-data.svg?branch=master)
[![Code Coverage](https://codecov.io/github/jonthemango/big-data/coverage.svg)](https://codecov.io/gh/jonthemango/big-data)

## Authors
Jon Mongeau - https://github.com/jonthemango
Ribal Aladeeb - https://github.com/ribal-aladeeb
Feras Younis - https://github.com/FerasYounis


# Abstract
Many people who lack an extensive credit history find it exceedingly difficult to receive a loan. As a result, these people get taken advantage of by shady lenders. The purpose of this project is to use alternative data about customers to determine whether or not they have the ability to repay their loans. This is a supervised binary classification problem. 

The dataset is provided by Kaggle (https://www.kaggle.com/c/home-credit-default-risk/overview) in the context of a competition. The training data contains 300K+ records of historical customer data labeled 0 or 1 for having repaid the loan or not. There are 122 available columns available to select our features from. The point of the project is to use this historical data to accurately predict if customers can repay loans without requiring their credit history.


# Introduction

## Context
[Home Credit](http://www.homecredit.net/about-us.aspx) aims to offer loan services to unbanked and underbanked people. Their target clientele could be either new immigrants, people of lower income, people recently got out of homelessness or anyone who simply has not had

## Objectives
The main goal is to find the best predictors to determine whether a client can repay a loan, given that they do not have a substantial credit history.

## Presentation of the Problem to Solve
The problem is a supervised binary classification learning problem.

## Related Work
- reference to other data

# Materials and Methods

## The Dataset
The entire dataset emcompasses 7 csv files.  

| File Name  | Shape | Description | Imfortant Features
|---|---|---|---|
| **data\application_train.csv**  | 307,511 rows, 122 columns  | Contains information about the approval of loan applications. Each row is a loan application. | `SK_ID_CURR` identifies the loan application. `TARGET` is a 0 if the loan was repaid while a 1 determines whether the loan is not repaid. |
| data\bureau.csv | 1,716,428 rows, 17 columns | Contains information about loan applicants' credit history from other banks and financial institutions. Each row is a credit.  | `SK_ID_BUREAU` identifies the credit. `SK_ID_CURR` identifies the loan application. |
| data\bureau_balance.csv | 27,299,925 rows, 3 columns  | Contains monthly data about credits from `bureau.csv`. Each row is a monthly credit. | `SK_ID_BUREAU` identifies the credit. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent) |
| data\previous_application.csv | 1,670,214 rows, 37 columns | Each row represents one previous loan application. There may be many previous loan applications for every current application in `application_train.csv`  | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application.  |
| data\POS_CASH_BALANCE.csv | 10,001,358 rows, 8 columns | Contains monthly data about previous POS and cash loans. Each row is a month of a loan. | `SK_ID_PREV` identifies the previous loan, `SK_ID_CURR` identifies the current application loan. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent) |
| data\credit_card_balance.csv | 3840312 rows, 23 columns | Contains monthly data about credit cards clients had with the company. Each row is a monthly credit card balance. | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent). |
| data\installments_payments.csv | 13605401 rows, 8 columns | Contains payment history data for previous loans applications. Each row is a payment made or missed.  | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application. `AMT_PAYMENT` identifies the amount payed (0.0 means missed). |

Additionally we are provided with `HomeCredit_columns_descriptions.csv` which provides an in-depth description for the column names in each table.

Preliminarily we wanted to do some pre-processing on the  **data\application_train.csv** data to see what percentage of instances were missing a particular feature. Our results for this can be found [in this outputted CSV](https://github.com/jonthemango/big-data/blob/master/preprocessing/missing_values.csv). We were able to determine that of the 122 columns, 55 columns were completely intact (meaning there were no records missing that feature). 

Another interesting thing to note about the available data is the distribution between loans which have been approved and loans which have not been approved.
```
>>> df = data.select("TARGET")
>>> not_approved = df.where(data["TARGET"] == 1).count() # Not been approved for a loan
>>> approved = df.where(data["TARGET"] == 0).count() # Approved for a loan
# 282,686 approved, 24,825 not approved
>>> approved/(approved+not_approved)
0.9192711805431351
```
Meaning only about 9.2 % of applications actually get approved for a loan. We will need to take this into consideration when classifying our data. One class of loan applications is far larger than the other. We will either need to sub-sample our data (after already doing so after seperating into train/test) or we will need to add linear combinations of approved loans. We will experiment with both approaches to see what makes our model most effective.

## Technologies and Algorithms
- Spark
- Spark ML: ML library got algorithms to work with features. these algorithms could be grouped into 4 areas.
Extraction, Transformation, Selection and LSH "Locality Sensitive Hashing".
- Spark RDD
- Pandas
- DataFrames
algorithms
- Algorithms?
- Clustering -> Divisive "top down" algorithm
- Clustering -> k-means algorithm
- CURE algorithm 
- User Profile and Prediction

- baseline using random and knn. Then possibly train a decision tree or random forest.