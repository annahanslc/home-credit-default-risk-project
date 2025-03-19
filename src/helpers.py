########################################################################
# CUSTOM HELPER FUNCTIONS
########################################################################

# Look for the src folders
# Need to only run once per environment

import sys
import os

# Get the path to the 'src' directory
src_path = os.path.abspath(os.path.join(os.getcwd(), '../src'))

# Add the 'src' folder to sys.path
if src_path not in sys.path:
  sys.path.append(src_path)

########################################################################
########################################################################
# Function for evaluating different regression models
########################################################################

from sklearn.metrics import (mean_absolute_error, mean_squared_error, root_mean_squared_error,
r2_score, mean_absolute_percentage_error)

import numpy as np

def adj_r2(r2, x):
  n = x.shape[0]
  p = x.shape[1]
  return 1 - (((n-1) / (n - p- 1)) * (1 - r2))

def evaluate_regression(model, X, y, name='model'):
  y_pred = model.predict(X)
  MAPE = mean_absolute_percentage_error(y, y_pred)
  r2 = r2_score(y, y_pred)
  RMSE = root_mean_squared_error(np.exp(y), np.exp(y_pred))
  MSE = mean_squared_error(y, y_pred)
  MAE = mean_absolute_error(y, y_pred)
  a_r2 = adj_r2(r2, X)
  metrics = ['MAE','MSE','RMSE','MAPE','R2','adj_r2']
  results = pd.DataFrame(columns=metrics, index=[name])
  results['MAE'] = [MAE]
  results['MSE'] = [MSE]
  results['RMSE'] = RMSE
  results['MAPE'] = MAPE
  results['R2'] = r2
  results['adj_r2'] = a_r2

  return results

########################################################################
########################################################################
# Custom transformer for dropping outliers
########################################################################

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
  def __init__ (self, columns, iqr_multiplier=1.5):
    """Calculate outliers based on IQR times a multiplier.

    column: list of columns to check for outliers
    iqr_multiplier: set the outlier range, defaults to 1.5"""

    self.columns = columns
    self.iqr_multiplier = iqr_multiplier

  def fit(self, X, y=None):
    #Calculate the IQR and thresholds for outlier detection.
    self.thresholds_ = {}

    for column in self.columns:
      if column not in X.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

      Q1 = X[column].quantile(0.25)
      Q3 = X[column].quantile(0.75)
      IQR = Q3-Q1

      lower_threshold = Q1 - (IQR * self.iqr_multiplier)
      upper_threshold = Q3 + (IQR * self.iqr_multiplier)

      self.thresholds_[column] = (lower_threshold, upper_threshold)

    return self

  def transform(self, X, y=None):
    #Remove outliers based on the calculated thresholds.
    X = X.copy()

    mask = pd.Series(True, index=X.index)

    for column, (lower_threshold, upper_threshold) in self.thresholds_.items():
      mask &= X[column].isna() | (X[column] >= lower_threshold) & (X[column] <= upper_threshold)

    return X[mask].reset_index(drop=True)


########################################################################
########################################################################
# Function for checking for outliers
########################################################################

def check_outliers(data, column, iqr_multiplier=1.5):
  # calculate 25% and 75% quantile
  qt_25 = data[column].quantile(0.25)
  qt_75 = data[column].quantile(0.75)

  # calculate the interquartile range
  iqr = qt_75 - qt_25

  # calculate the lower and upper thresholds
  lower = qt_25 - iqr*iqr_multiplier
  upper = qt_75 + iqr*iqr_multiplier

  data_wo_outliers = data[(data[column] >= lower) & (data[column] <= upper)]
  data_outliers = data[(data[column] < lower) | (data[column] > upper)]

  num_outliers = data_outliers[column].count()
  num_data = data[column].count()

  percent_outlier = '{:.2%}'.format(num_outliers / num_data)


  print(f'The original dataframe contains {num_data} observations.')
  print(f'Using IQR * {iqr_multiplier}, {num_outliers} outliers were detected.')
  print(f'If removed, {percent_outlier} of the data will be dropped.')


########################################################################
# Function for evaluation classification models
########################################################################

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report,
                             ConfusionMatrixDisplay, roc_auc_score)


def eval_classification(model, X_train, y_train, X_test, y_test, model_name='model', results_frame=None, pos_label=1,
                        average='binary', roc_auc_avg='macro'):

  model.fit(X_train, y_train)
  train_pred = model.predict(X_train)
  test_pred = model.predict(X_test)

  print('Train Evaluation')
  print(classification_report(y_train, train_pred))
  ConfusionMatrixDisplay.from_predictions(y_train, train_pred, normalize='true', cmap='Blues')
  plt.show()

  print('Test Evaluation')
  print(classification_report(y_test, test_pred))
  ConfusionMatrixDisplay.from_predictions(y_test, test_pred, normalize='true', cmap='Greens')
  plt.show()

  results = pd.DataFrame(index=[model_name])
  results['train_accuracy'] = accuracy_score(y_train, train_pred)
  results['test_accuracy'] = accuracy_score(y_test, test_pred)
  results['train_precision'] = precision_score(y_train, train_pred, pos_label=pos_label, average=average)
  results['test_precision'] = precision_score(y_test, test_pred, pos_label=pos_label, average=average)
  results['train_recall'] = recall_score(y_train, train_pred, pos_label=pos_label, average=average)
  results['test_recall'] = recall_score(y_test, test_pred, pos_label=pos_label, average=average)
  results['train_f1'] = f1_score(y_train, train_pred, pos_label=pos_label, average=average)
  results['test_f1'] = f1_score(y_test, test_pred, pos_label=pos_label, average=average)
  results['train_auc'] = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1], average=roc_auc_avg, multi_class='ovr')
  results['test_auc'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], average=roc_auc_avg, multi_class='ovr')

  if results_frame is not None:
    results = pd.concat([results_frame, results])

  return results, train_pred, test_pred
