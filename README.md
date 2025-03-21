# Home Credit - Reducing Loan Default Risk

![houses_cute](https://github.com/user-attachments/assets/1bc625d1-d0f7-4a85-b6f7-88ea5a3c7ceb)



### Project Objective

Home buyers who have little or insufficient credit history struggle to obtain loans. By using alternative information alongside traditional credit information, my goal is to produce a reliable model to help predict home loan default. The ultimate objective is to meet the needs of the underserved market while minimizing default risk for the company.

### Directory 

1. About the Data
2. Feature Engineering
3. Preprocessing Pipeline
4. Model Selection
5. Feature Selection
6. Analysis
7. Next Steps


### About the Data

The original datasets can be found here: [https://www.kaggle.com/competitions/home-credit-default-risk/data]

The data is spread across 7 different tables. The main table is the application data. The other tables are connect to the main table through the loan identifier SK_ID_CURR, or through an intermediary identifier SK_ID_BUREAU or SK_ID_PREV.

Each SK_ID_CURR can be connected to multiple SK_ID_BUREAU or SK_ID_PREV, which in turn can be associated with multiple records. In order for these values to be merged into the main dataset for model training, they have to be consolidated down to one value per feature per SK_ID_BUREAU. 

### Feature Engineering

Prior to merging the data together, I reviewed each spreadsheet for features that could be helpful in my model. Mainly, I looked for statistics that might indicate credit worthiness, both positive and negative indicators. I then engineered these features by performing a groupby and aggregating the value, this could be either the mean, median, min or max. 

Examples of engineered features include:

From bureau table:

<img src="https://github.com/user-attachments/assets/6ab29a9e-c232-490f-aa2c-67aa5d9ce56c" width="800" />

###

##### For all engineered features broken down by the table they were derived from, see [the engineering folder](https://github.com/annahanslc/home-credit-default-risk-project/tree/main/engineering)


### Preprocessing Pipeline

My preprocessing pipeline includes the following steps:

1. **Feature cleaning:** 'DAYS_EMPLOYED' are mostly negative values because the number indicates for how many previous days have they been employed at their current employer, relative to the date of the application. However, there are 55,374 observations that have a positive value of 365243, furthermore, there are no other values in between 0 and 365243. This indicates that '365243' is actually a placeholder value. This makes me wonder how the data is accounting for customers who are not employed at all. Checking the frequency of the value 0, there are only 2 observations. This confirms my hypothesis that '365243' is inputed for those who are not emloyed. Since DAYS_EMPLOYED is a numeric discrete feature, the numeric correlations between values are important, but this placeholder value will skew the data significantly and obscure the numeric relationships between real datapoints. In order to preserve the numeric significance of those with job (negative numbers, relative to the date of their application), but remove the skew caused by 365243, I will replace DAYS_EMPLOYED's placeholder value of 365243 with 365

2. **Feature filtering:** This proprocessor functions as a filter to filter out unexpected columns. It also helps to organize which features are original to the dataset, and which ones were engineered. As I conduct feature selection, the filters helps me ensure that only the selected features remain in the pipeline.

3. **Imputing:** All features, original and engineered, are equipped with an imputation method. This will safeguard against nulls in new, incoming data. For categorical features, they are mostly imputed using the most frequent value. For numeric features, they are imputed either with 0, or with the median.

4. **Encoding:** Categorical features are encoded using the OneHotEncoder. It is set to drop one out of the two columns if it is a binary feature, and require a minimum frequency of 100, which means that all values that have fewer than 100 observations are combined into a separate column for infrequent values. 

5. **Log Tranformer:** Many models, include logistic regression, SVM's and KNN, assume that data is normally distributed. A skewed distribution will mask the true linear relationship between features and the target variable. In order to help the models better to detect linear patterns in the data, I will use log transformation to make skewed distributions more normal. 

6. **Scaling:** Numeric, non-OneHotEncoded, features are scaled using the StandardScaler. The StandardScaler standardizes features so that it has a mean of 0 and standard deviation of 1.

   
### Model Selection

1. **Addressing Target Imbalance:** The dataset contains 8% of the positive class, which is loan default, and 92% of the negative class, which is non-default. This imbalance will encourage models to overly predict the negative class, which means that the model is incentivized to predict that a customer will *not* default. This is counterproductive to the goals of my model, as the objective is to reduce risk of loan-default, which means the model needs to vigilently predict defaults. In order to reduce this imbalance, I will use undersampling, balanced class weight, SMOTE, and statifying the train test split.

2. Main models performance:


|                      Model |   train_accuracy |   test_accuracy |   train_precision |   test_precision |   train_recall |   test_recall |   train_f1 |   test_f1 |   train_auc |   test_auc |
|---------------------------:|-----------------:|----------------:|------------------:|-----------------:|---------------:|--------------:|-----------:|----------:|------------:|-----------:|
|       LogReg Undersampling |         0.614923 |        0.618799 |          0.131898 |         0.133275 |       0.675428 |      0.676334 |   0.220698 |  0.222672 |    0.693414 |   0.698263 |
|            LogReg Balanced |         0.641012 |        0.646668 |          0.137057 |         0.139472 |       0.650806 |      0.653172 |   0.226429 |  0.229861 |    0.695179 |   0.700966 |
| DecisionTree leaf=100 d=10 |         0.681819 |        0.677577 |          0.15658  |         0.15035  |       0.670544 |      0.643706 |   0.253877 |  0.243765 |    0.738248 |   0.718206 |
|               XGB Balanced |         0.726086 |        0.723168 |          0.186018 |         0.178185 |       0.708862 |      0.672508 |   0.294701 |  0.281725 |    0.794591 |   0.768486 |

Other models:

|                      Model |   train_accuracy |   test_accuracy |   train_precision |   test_precision |   train_recall |   test_recall |   train_f1 |   test_f1 |   train_auc |   test_auc |
|---------------------------:|-----------------:|----------------:|------------------:|-----------------:|---------------:|--------------:|-----------:|----------:|------------:|-----------:|
|   RandomForest d=12, n=500 |         0.686596 |        0.678162 |          0.155255 |         0.136704 |       0.648993 |      0.561934 |   0.250569 |  0.219910 |    0.736374 |   0.672979 |
|          LogReg with SMOTE |         0.665698 |        0.663602 |          0.143105 |         0.144358 |       0.629249 |      0.630714 |   0.233180 |  0.234943 |    0.703414 |   0.703288 |



### Feature Selection


### Analysis


### Next Steps


### References:

Banner Image from <a href="https://www.freevector.com/pastel-abstract-geometric-houses-background-71471">FreeVector.com</a>
