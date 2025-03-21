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

The data is spread across 7 different tables. The main table is the application data and contains 307,511 observations. The other tables are connect to the main table through the loan identifier SK_ID_CURR, or through an intermediary identifier such as SK_ID_BUREAU or SK_ID_PREV.

Each SK_ID_CURR can be connected to multiple SK_ID_BUREAU or SK_ID_PREV, which in turn can be associated with multiple records. In order for these values to be merged into the main dataset for model training, they have to be consolidated down to one value per feature per SK_ID_BUREAU. 

### Feature Engineering

Prior to merging the data together, I reviewed each spreadsheet for features that could be helpful in my model. Mainly, I looked for statistics that might indicate credit worthiness, both positive and negative indicators. I then engineered these features by performing a groupby and aggregating the value, this could be either the mean, median, min or max. 

Examples of engineered features include:

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

2. **Main Models:**


|                      Model |   train_accuracy |   test_accuracy |   train_precision |   test_precision |   train_recall |   test_recall |   train_f1 |   test_f1 |   train_auc |   test_auc |
|---------------------------:|-----------------:|----------------:|------------------:|-----------------:|---------------:|--------------:|-----------:|----------:|------------:|-----------:|
|       LogReg Undersampling |         0.614923 |        0.618799 |          0.131898 |         0.133275 |       0.675428 |      0.676334 |   0.220698 |  0.222672 |    0.693414 |   0.698263 |
|            LogReg Balanced |         0.641012 |        0.646668 |          0.137057 |         0.139472 |       0.650806 |      0.653172 |   0.226429 |  0.229861 |    0.695179 |   0.700966 |
| DecisionTree leaf=100 d=10 |         0.681819 |        0.677577 |          0.15658  |         0.15035  |       0.670544 |      0.643706 |   0.253877 |  0.243765 |    0.738248 |   0.718206 |
|               XGB Balanced |         0.726086 |        0.723168 |          0.186018 |         0.178185 |       0.708862 |      0.672508 |   0.294701 |  0.281725 |    0.794591 |   0.768486 |

3. **Other Models:**

|                      Model |   train_accuracy |   test_accuracy |   train_precision |   test_precision |   train_recall |   test_recall |   train_f1 |   test_f1 |   train_auc |   test_auc |
|---------------------------:|-----------------:|----------------:|------------------:|-----------------:|---------------:|--------------:|-----------:|----------:|------------:|-----------:|
|   RandomForest d=12, n=500 |         0.686596 |        0.678162 |          0.155255 |         0.136704 |       0.648993 |      0.561934 |   0.250569 |  0.219910 |    0.736374 |   0.672979 |
|          LogReg with SMOTE |         0.665698 |        0.663602 |          0.143105 |         0.144358 |       0.629249 |      0.630714 |   0.233180 |  0.234943 |    0.703414 |   0.703288 |

4. **Final Model:**

The model I chose to move forward with is the XGBoost with balanced class weights. 
The confusion matrices for this model are as follows. The scores are normalized on the true axis, which means that the numbers add up to 1 horizontally.

![model_perf_train](https://github.com/user-attachments/assets/b258ef88-cfc2-4ff2-8c20-9c1f26476520)
![model_perf_test](https://github.com/user-attachments/assets/dbab50b0-f527-4bba-9e96-72a23fa6ef56)

Model performance analysis:

Based on the test data, my model will incorrectly predict default (False Positive) 27% of the time, and miss a true default (False Negative) 33% of the time. With the purpose of reducing loan default risk, false negatives are more costly than false positives, because failing to predict a true default means that the company will suffer losses up to the value of the loan, with could be hundreds of thousands of dollars, while incorrectly predicting a loan will default will only result in the company losing a potential client. The tradeoff is not one-to-one.

Although not ideal, the model is able to accurately predict a default 67% of the time, which is a promising. I will continue to improve the model through feature selection.


### Feature Selection

Between the original dataset and engineered features, I had 230 features ready for preprocessing. To better understand their impact on the model, I started with only 43 features, mainly from the original dataset, and added the other features in small groups of 2-5 features at the time. If my 4 main models showed an uplift in performance, I would keep the added features, otherwise, they were dropped. Through this process, I selected 84 final features to enter the preprocessing pipeline. 

After preprocessing, the processed dataset contained 134 features. These were the features that the models were trained on during model selection. 

To further improve feature selection, I will first check for multicollinearity, and then filter using permutation importance.

1. **Removing Features with High Multicollinearity**

Using SelectNonCollinear with a threshold of 0.7, the 134 features were checked for multicollinearity, and 15 features were dropped due to high correlation with another feature.

2. **Filtering Features using Permutation Importance**

The remaining 119 features are then checked for their importance in the main model, XGBoost with balanced weights, using permutation importance. Due to the large dataset, the data is resampled to obtain a random subset of 10,000 samples. This helps to greatly reduce the processing time. Once the feature importances were calculated, I created a filter to create a boolean mask that only passes through features with an importance of 0.001 or higher. 

The above importance mask narrowed down the features from 119 to 30 features. This means that fewer than 1/3 of the feature remain. 

The performance of the model after feature selection is as below:

![model_filtered_train](https://github.com/user-attachments/assets/d4a61965-0134-449d-9edb-438c39a395c9) 
![model_perf_test](https://github.com/user-attachments/assets/9754a85f-c181-4ac3-a644-3ba945bff827)

###
|                 |   XGB w/134 features |   XGB w/30 features |      Uplift |
|:----------------|---------------------:|--------------------:|------------:|
| train_accuracy  |             0.726086 |            0.72194  | -0.00414621 |
| test_accuracy   |             0.723168 |            0.718436 | -0.00473148 |
| train_precision |             0.186018 |            0.183139 | -0.00287859 |
| test_precision  |             0.178185 |            0.176174 | -0.00201004 |
| train_recall    |             0.708862 |            0.706395 | -0.00246727 |
| test_recall     |             0.672508 |            0.676737 |  0.00422961 |
| train_f1        |             0.294701 |            0.290868 | -0.0038327  |
| test_f1         |             0.281725 |            0.279569 | -0.00215561 |
| train_auc       |             0.794591 |            0.789211 | -0.00538037 |
| test_auc        |             0.768486 |            0.765326 | -0.00315918 |

###

**Feature selection analysis:** Although the only uplift in scoring was test_recall, the other scores only decreased minimally. At the same time, the drastic reduction in the number of features greatly benefits processing time and cuts down on computing. 

### Analysis

**1. The 30 features ordered by importance:**

![8_feat_importances](https://github.com/user-attachments/assets/a5a9a72a-13d7-4694-a173-b97a72b83717)

Observations based the feature importances:
- The 3 external sources, which are the scores from the 3 major credit bureaus, are the most important features to this model. This confirms that traditional credit evaluation metrics are effective and should always be used when available. 
- DAYS_BIRTH and CODE_GENDER_M raise questions about discrimination and should be recommended to be removed from the model
- Previous application decisions seem to drive future predictions, features that relate to prior financial obligations include:
    - prev_avg_ratio_credit_approved: the ratio of credit applied to credit approved on previous applications
    - prev_status_approved: count of approved previous application
    - prev_status_refused: count of refused previous applications
    - prev_yield_high: when a previous application was deemed to require a high interest rate, which can indicate higher risk
    - prev_yield_low_action: this is when a previous application was provided a low interest rate, not necessarily due to credit worthiness, but due to some type of promotion the company was offering at that time
- Credit Card utilization behavior is utilized by the model to make predictions, these features include:
    - avg_cc_cnt_ATM_drawings: the average number of ATM drawings per month per credit card, averaged per client
    - cc_avg_credit_usage_ratio: the average ratio of credit card balance to credit card limit
- Being employed is also a major factor that the model considers. These include the following features:
    - DAYS_EMPLOYED
    - NAME_INCOME_TYPE_Working
- Employment is also related to stability, which ties to a few other features that the model found indicative:
    - DAYS_REGISTRATION: this is how long ago did the client change their registration
    - DAYS_ID_PUBLISH: this is how long ago did the client change the identity document that was used to apply for the loan
    - DAYS_LAST_PHONE_CHANGE: this is how long ago did the client last change their phone number

### 

**2. Plotting the SHAP Plot to see how the Model uses Features to make Predictions**

![shap_plot_30](https://github.com/user-attachments/assets/06879d4d-e26b-4081-8f62-60d43901e459)

### 

**Analysis of the SHAP Plot:**

- As expected, lower credit bureau scores drive predictions towards default
- Larger loans are predicted to be more likely to default
- The model is biased against male applications, this raises a concern of discrimination. Feature should be removed.
- The model utilizes age in its prediction, but it does not appear that the relationship is linear across age. Regardless, this features needs to be reconsidered due to discrimination.
- Not owning a car is a driver of default in this model. I will check the correlation of the raw data to see if the correlation shows similar tendencies.

**3. Analysis of Correlations**

1. DAYS_EMPLOYED vs TARGET: The target, default, is correlated with a shorter employment period.

      ![days_employed_box_transparent](https://github.com/user-attachments/assets/a1e2a32c-1e59-4130-95f2-d506d97d8109)


2. cc_avg_credit_usage_ratio vs TARGET: A higher average credit card balance to limit ratio is notably more correlated with default

      ![cc_avg_credit_usage_ratio_transparent](https://github.com/user-attachments/assets/6fe349ce-9979-4591-8037-c01b07120806)


3. avg_cc_cnt_ATM_drawings: The average number of ATM drawings per credit card per month correlates more ATM drawings to more likelihood of default.

      ![avg_num_ATM_transparent](https://github.com/user-attachments/assets/aa3083da-1df4-4d35-8f7d-66e8b6b67bd7)


4. NAME_CONTRACT_TYPE: Cash loans are more likely to default than revolving loans.

      ![contract_type_transparent](https://github.com/user-attachments/assets/7f87366f-7f37-4e7a-bfe3-5fc43ae0ae1b)

5. FLAG_OWN_CAR: Clients who own a car has a lower rate of default in this dataset. This makes sense, because purchasing a vehicle also requires obtaining a loan. If the client was approved for a car loan, they are more likely to qualify for a home loan than someone who was not approved for a car loan.

      ![own_car_transparent](https://github.com/user-attachments/assets/4431834a-1127-49f3-8a20-fb96cc5f7d98)


### Next Steps


### References:

Banner Image from <a href="https://www.freevector.com/pastel-abstract-geometric-houses-background-71471">FreeVector.com</a>
