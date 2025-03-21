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


### Feature Selection


### Analysis


### Next Steps


### References:

Banner Image from <a href="https://www.freevector.com/pastel-abstract-geometric-houses-background-71471">FreeVector.com</a>
