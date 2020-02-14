![Build Status](https://travis-ci.org/jonthemango/big-data.svg?branch=master)
[![Code Coverage](https://codecov.io/github/jonthemango/big-data/coverage.svg)](https://codecov.io/gh/jonthemango/big-data)

## Authors
Jon Mongeau - https://github.com/jonthemango
Ribal Aladeeb - https://github.com/ribal-aladeeb


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
| data\application_train.csv  | 307,511 rows, 122 columns  | Contains information about the approval of loan applications. Each row is a loan application. | `SK_ID_CURR` identifies the loan application. `TARGET` is a 0 if the loan was repaid while a 1 determines whether the loan is not repaid. |
| data\bureau.csv | 1,716,428 rows, 17 columns | Contains information about loan applicants' credit history from other banks and financial institutions. Each row is a credit.  | `SK_ID_BUREAU` identifies the credit. `SK_ID_CURR` identifies the loan application. |
| data\bureau_balance.csv | 27,299,925 rows, 3 columns  | Contains monthly data about credits from `bureau.csv`. Each row is a monthly credit. | `SK_ID_BUREAU` identifies the credit. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent) |
| data\previous_application.csv | 1,670,214 rows, 37 columns | Each row represents one previous loan application. There may be many previous loan applications for every current application in `application_train.csv`  | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application.  |
| data\POS_CASH_BALANCE.csv | 10,001,358 rows, 8 columns | Contains monthly data about previous POS and cash loans. Each row is a month of a loan. | `SK_ID_PREV` identifies the previous loan, `SK_ID_CURR` identifies the current application loan. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent) |
| data\credit_card_balance.csv | 3840312 rows, 23 columns | Contains monthly data about credit cards clients had with the company. Each row is a monthly credit card balance. | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent). |
| data\installments_payments.csv | 13605401 rows, 8 columns | Contains payment history data for previous loans applications. Each row is a payment made or missed.  | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application. `AMT_PAYMENT` identifies the amount payed (0.0 means missed). |

Additionally we are provided with `HomeCredit_columns_descriptions.csv` which provides an in-depth description for the column names in each table.



## Technologies and Algorithms
- Spark
- Spark ML
- Algorithms?

- baseline using random and knn. Then possibly train a decision tree or random forest.