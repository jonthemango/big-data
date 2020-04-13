## Authors
[Ribal Aladeeb](https://github.com/ribal-aladeeb)

[Jon Mongeau](https://github.com/jonthemango)

[Feras Younis](https://github.com/FerasYounis)

## Navigating Repository
`/src/preprocessing/` contains a set of pre-processing steps including data exploration, feature engineering and undersampling.

`/src/runners/` contains a set of drivers for various splitting into train/test, training models using algorithms. `random_forest`

`/src/evaluators` contains code where we define our own precision, recall and F1 evaluator.

`/notebooks/` contains a set of notebooks used for data exploration

#### Try it locally
```
# If you don't have Spark 2.4.5
pip install -r requirements.txt

# to train a model on a small subsample of data
python runner.py random_forest sample
```

# Abstract
Using labeled data from individual home credit applications we created a binary classifier in order to determine if a given application would have trouble repaying their loan or not. The [dataset](https://www.kaggle.com/c/home-credit-default-risk/overview) is provided by Kaggle for a competition. The training data contains 300K+ records of historical customer data labeled 0 or 1 for having had trouble repaying the loan or not. We use this historical data to predict if customers can repay loans without requiring their credit history.

We follow a supervised learning approach to classifying instances. We conduct a comparative analysis of the effectiveness of linear support-vector machine, decision tree and random forest when used to train our models. A variety of techniques are employed for feature engineering including one-hot encoding, string indexing, and encoding aggregate data into statistics.


# Introduction

## Context
[Home Credit](http://www.homecredit.net/about-us.aspx) aims to offer loaning services to unbanked and underbanked people. Their target clientele could be either new immigrants, people of lower income, people who recently got out of homelessness or anyone who simply has not had access to banking services. The company posted this dataset of historical loans that they have given out along with all the information that they collect regarding those loans in the context of a kaggle competition. The competition is now expired but you can still see the top submissions and there is some good documentation on where to get started.

## Objective
The main goal is to find the best predictors to determine whether a client can repay a loan, given that they do not have a substantial credit history. The company provides an extensive amount of data regarding their customers across multiple tables. Our job is to find a winning combination of features across all tables that will yield the best precision, recall, and area under ROC possible. The winning score on kaggle is 0.80570 using area under the ROC curve.

## Presentation of the Problem to Solve
This project is a supervised binary classification learning problem. Our first problem is that a lot of columns have missing values, some are very sparse. For this we wrote a quick [script](https://github.com/jonthemango/big-data/blob/master/src/preprocessing/step1_column_analysis.py) that counts the number of missing values and the number of unique values by column for each table in order to identify numerical and categorical features. That script generates a report which summarizes this information in both [json](https://github.com/jonthemango/big-data/blob/master/src/preprocessing/missing_values.json) and [csv](https://github.com/jonthemango/big-data/blob/master/src/preprocessing/missing_values.csv) format.

Furthermore, we have a combination of about 50M+ records across 7 tables. The common denominator is a column named SK_ID_CURR which denotes the ID of a current loan application. This ID is unique in the `application_train.csv` file but can be repeated in other tables, which creates multiplicity. A big challenge is to aggregate the data related to a given SK_ID_CURR from all tables into a single feature vector for training.

Another major problem to solve is the class imbalance problem. About 92% of our 300K+ instances are labeled 0 (loan is repaid on time). We will need to take this into consideration when classifying our data. One class of loan applications is far larger than the other. Our approach will be to subsample the majority class.
```
>> df = data.select("TARGET")
>> late = df.where(data["TARGET"] == 1).count()
>> ontime = df.where(data["TARGET"] == 0).count()

>> print(ontime, "on time,",late, "late")

>> percent_ontime = round(ontime/(ontime+late)*100,2)
>> print(f'{percent_ontime}% of loans are repaid on time')

282686 on time, 24825 late
91.93% of loans are repaid on time
```

## Related Work

In Will Koehrsen's notebook [Introduction to Dataset/Problem](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction), Koehrsen explores the dataset and does a good preliminary analysis of some important columns. He performs a correlation between various feature columns and the target using the Pearson correlation coefficient. He identifies a list of positively correlated columns and a list of negatively correlated columns. When conducting our own feature engineering we made use of this notebook often in order to identify features. We also consulted heavily his other notebooks on feature engineering including [Introduction to Manual Feature Engineering](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering) and [Automated Feature Engineering basics](https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics).


# Materials and Methods

## The Dataset
The entire dataset emcompasses 7 csv files. For the purposes of our classification problem we generated features from the first two tables in bold. The reason for only using these two is training time.

| File Name  | Shape | Description | Important Features
|---|---|---|---|
| **[application_train.csv](https://github.com/jonthemango/big-data/blob/master/data/sample/sample_application_train.csv)**  | 307,511 rows, 122 columns  | Contains information about the approval of loan applications. Each row is a loan application. | `SK_ID_CURR` identifies the loan application. `TARGET` is a 0 if the loan was repaid while a 1 determines whether the loan is not repaid. |
| **[bureau.csv](https://github.com/jonthemango/big-data/blob/master/data/sample/sample_bureau.csv)** | 1,716,428 rows, 17 columns | Contains information about loan applicants' credit history from other banks and financial institutions. Each row is a credit.  | `SK_ID_BUREAU` identifies the credit. `SK_ID_CURR` identifies the loan application. |
| [bureau_balance.csv](https://github.com/jonthemango/big-data/blob/master/data/sample/sample_bureau_balance.csv) | 27,299,925 rows, 3 columns  | Contains monthly data about credits from `bureau.csv`. Each row is a monthly credit. | `SK_ID_BUREAU` identifies the credit. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent) |
| [previous_application.csv](https://github.com/jonthemango/big-data/blob/master/data/sample/sample_previous_application.csv) | 1,670,214 rows, 37 columns | Each row represents one previous loan application. There may be many previous loan applications for every current application in `application_train.csv`  | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application.  |
| [POS_CASH_balance.csv](https://github.com/jonthemango/big-data/blob/master/data/sample/sample_POS_CASH_balance.csv) | 10,001,358 rows, 8 columns | Contains monthly data about previous POS and cash loans. Each row is a month of a loan. | `SK_ID_PREV` identifies the previous loan, `SK_ID_CURR` identifies the current application loan. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent) |
| [credit_card_balance.csv](https://github.com/jonthemango/big-data/blob/master/data/sample/sample_credit_card_balance.csv) | 3,840,312 rows, 23 columns | Contains monthly data about credit cards clients had with the company. Each row is a monthly credit card balance. | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application. `MONTHS_BALANCE` represents the number of months relative to the application date (-1 means most recent). |
| [installments_payments.csv](https://github.com/jonthemango/big-data/blob/master/data/sample/sample_installments_payments.csv) | 13,605,401 rows, 8 columns | Contains payment history data for previous loans applications. Each row is a payment made or missed.  | `SK_ID_PREV` identifies the previous loan application. `SK_ID_CURR` identifies the current loan application. `AMT_PAYMENT` identifies the amount payed (0.0 means missed). |

## Technologies and Algorithms

We use Apache Spark Dataframes and RDDs for pre-processing and Spark ML for training models. We are interested in applying algorithms like Linear Support Vector Machines, Decision Tree and Random Forest. We experimented with kNN but found it was computationally infeasible.

### Feature Engineering
For our analysis we built 2 distinct feature vectors. The methods used to generate those vectors are available [here](https://github.com/jonthemango/big-data/blob/60d7b98b49b1addb298c3873ddeed230eff4a92a/src/preprocessing/step2_feature_engineering.py#L105) and [here](https://github.com/jonthemango/big-data/blob/60d7b98b49b1addb298c3873ddeed230eff4a92a/src/preprocessing/step2_feature_engineering.py#L194). The results pertain to the second one, as the first yielded extremely low precision and recall due to lack of true positives. The false positive rate was roughly 10 times greater than the true positive rate and the false negative rate was hundreds, sometimes thousands of times greater than the true positive rate.

We employed a variety of techniques for encoding features into our feature vector. For categorical variables we employed the use of [one-hot encoding](https://github.com/jonthemango/big-data/blob/60d7b98b49b1addb298c3873ddeed230eff4a92a/src/preprocessing/step2_feature_engineering.py#L32). For binary categorical variables we used [string indexing](https://github.com/jonthemango/big-data/blob/60d7b98b49b1addb298c3873ddeed230eff4a92a/src/preprocessing/step2_feature_engineering.py#L53) to convert string based categories into binary floats. For any multiplicity where a table had several rows that reference the training dataset, we [encode the aggregation](https://github.com/jonthemango/big-data/blob/60d7b98b49b1addb298c3873ddeed230eff4a92a/src/preprocessing/step2_feature_engineering.py#L66) of those rows using statistics like mean, min and max. For float or integer variables we simply cast the string representation to the appropriate spark type. In the original kaggle competition, participants have posted their own notebooks for feature engineering in which they compute the columns most highly correlated with the 'TARGET' column. We used this information to choose our first features.

Then, we turned to the `bureau.csv` file. As mentioned earlier, this file contains a history of previous loan applications from the same applicants found in `application.csv`. We decided to calculate the ratio of previous `on-time/late loans` and build a feature out of it. Our intuition led us to believe that this could potentially be a good predictor for the present loan applications. We constructed another feature called `credit_over_income` which is a ratio of the loan's amount over the applicant's yearly annual income.

### Data Imbalance
Before implementing undersampling, none of our models yielded any positive predictions. Essentially, the classifier predicted all records to be on-time loans. To solve this, we found the simplest solution was to [undersample](https://github.com/jonthemango/big-data/blob/166244de44f3a1ffd805c111bd44989cc54f9804/src/preprocessing/step3_undersampling.py#L17) the majority class until a desired `class_ratio` was achieved. Generating linear combinations of the minority class is very complicated to implement in this dataset because of its sheer size, its multiplicity between tables, and the combination of numerical and categorical variables that form the feature vector.

### Algorithm Selection
We implemented four algorithms for training our models, three of which were provided by Spark ML and one we implemented ourselves (kNN). The specification and reasoning for those algorithms are provided in the table below.
|State| Specification | Tuning Explanation |
|---|---|---|
| [Linear SVM](https://github.com/jonthemango/big-data/blob/master/src/runners/linear_support_vector_machine.py) | maxIter=10, regParam=0.1 | Taken from Spark ML docs |
| [Decision Tree](https://github.com/jonthemango/big-data/blob/master/src/runners/decision_tree.py) | maxDepth=5, maxBins=32 | Left as Spark Default |
| [Random Forest](https://github.com/jonthemango/big-data/blob/master/src/runners/random_forest.py) | numTrees=100, maxDepth=10 | Doubled depth of default and increased numTrees until runtime became unreasonable. |
| [kNN](https://github.com/jonthemango/big-data/blob/master/src/runners/knn.py) | distanceFunction=Euclidean, k=5 | Since kNN was computationally infeasible for us (even on a sample), we never experimented with other distance functions. k=5 was selected during testing. |

### Big Data Infrastructure
We use Google's Dataproc for issuing spark jobs to a virtual cluster. Our cluster contains a single master node at 16 vCPU with 60Gb ram, with no other physical workers. We submitted about 140 jobs in total. This allowed us to train models in under 5 minutes.

# Results

### Evaluation
We define late loans (labeled 1) to be the positive class, and loans repaid on time (labeled 0) to be the negative class. Our negative class is of course the majority class. We split our training and testing data into 80% and 20% respectively. We had a plethora of things to tune like finding a good undersampling ratio, finding optimal hyperparameter values, 200+ possible features to choose from, multiple different algorithms to train, etc. We had to train a model each time in order to determine which recipe worked best (given that we have no prior ML experience). Therefore, we decided not to perform k-fold cross validation because training k times to evaluate our hypotheses was not feasible.

#### Confusion Matrix
|State|Label|Prediction|
|---|---|---|
|True Positive|1|1|
|True Negative|0|0|
|False Negative|1|0|
|False Positive|0|1|

Using spark's built-in evaluation module was not an option for precision, recall, and F1 scores as the numbers reported were very high compared to area under ROC. We suspect that spark inversed the meaning of positive/negative class and therefore the metrics were meaningless. We also did not want to use accuracy as a metric since accuracy can be very misleading: our initial classifier only predicted 0's and that yields a 92% accuracy, even though the classifier was useless.

To remediate this problem, we manually count the TP, TN, FP, and FN rates and compute precision, recall, and F1 using their respective formulas. We also used spark's area under ROC curve evaluator due to that metric being used as a baseline in the competition. So area under ROC was our only way to compare ourselves to other teams who attempted to solve the same problem.

### Best Runs

We achieved these results with the hyper-parameters presented in Algorithm Selection and an undersampling ratio of ~0.55-0.66 (i.e. the majority class makes up ~55-60% of the training set).
|**Algorithm**|**Area Under ROC**|**Precision**|**Recall**|**F1**
|---|---|---|---|---|
|SVM|0.72416|0.3448275|0.002452|0.0048685
|Decision Tree|0.39103|0.15314|0.585425|0.2427769
|**Random Forest**|**0.7285473**|**0.14728129**|**0.663621**|**0.24106221**
|**Competition winners**|**0.80570**|-|-|-

# Discussion
The usage of our solution as a binary classifier for determining whether a loan applicant will have trouble repaying their loan is likely not suitable for production use in a bank or financial institution. Our results are too low and our models (in their current state) would not be a useful tool to solve the problem effectively. Creditors would prefer to never approve a loan of someone who may face difficulties (false negative) and our solution does not provide this assurance.

One limitation we faced was in implementing the kNN algorithm. We have mentioned in this report that our implementation is computationally infeasible. This is due to the cartesian-product matrix of test and train set resulted in an `rdd` of about 14 billion rows `0.8*300k*0.2*300k`. We feel that our implementation of the algorithm could be improved and would likely be quite competitive when compared to other algorithms or at least useful as a baseline. We might also consider, just for this algorithm, in switching to sk-learn as they have an implementation available out of the box.

Another major challenge of this project was in implementing systems and tools that we could use to build our models. For example we implemented ourselves a set of spark wrapper methods for encoding various feature columns, as well as an undersampling module, as well as implementing an evaluator module for determining precision and recall (and F1). All this to say that we spent time developing tools to aid us rather than working on relevant features from other tables that we had.

Improving the usefulness of our feature engineering is a major next step for improving the overall quality of our models. Within the scope of this project we only managed to extract a small set of useful features. In the future, our goal would be to improve feature engineering by employing algorithms like generating a polynomial combination of features or even employing automated feature engineering.
