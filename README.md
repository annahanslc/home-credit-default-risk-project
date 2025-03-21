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

![image](https://github.com/user-attachments/assets/6ab29a9e-c232-490f-aa2c-67aa5d9ce56c)

For all engineered features broken down by the table they were derived from, see the engineering folder: [[engineering](https://github.com/annahanslc/home-credit-default-risk-project/tree/main/engineering)]


### Preprocessing Pipeline


### Model Selection


### Feature Selection


### Analysis


### Next Steps


### References:

Banner Image from <a href="https://www.freevector.com/pastel-abstract-geometric-houses-background-71471">FreeVector.com</a>
