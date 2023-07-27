# RESUME_CLASSIFIER
# A Multi-Class Classification NLP Project

## Introduction

Natural Language Processing (NLP) has gained popularity for multiple reasons and it is an exciting technology that is here to stay for a long time. NLP deals with machines understanding the way humans speak and write the language in their everyday lives. In this repository, I am going over one of the simple projects of that kind: classifying an applicant’s resume.

The conventional techniques of hiring a candidate for a position is becoming more labor intensive, therefore inefficient, because of the growing online recruitment. The companies receive an excessive number of resumes in multiple categories for the vacant positions.

Using some of the NLP and Machine Learning (ML) techniques, categorizing the applicants’ resumes for the available positions can be automated. In this repo, I developed a simplified version of such a multiclass classification in Python using NLP.

## Project Approach

### 1. Importing and Installing Necessary Libraries

I have imported necessary libraries like numpy, pandas, sklearn, nltk etc to use in the code block

### 2. Uploading and reading the csv file

I acquired the data from the below link and uploaded it as a pandas dataframe.
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

### 3. Generic Cleaning of Data

I have used functions like null_values, dtypes, drop_duplicates to do general overall cleaning of the data

### 4. Encoding and analysing distribution of target column

There are 24 different categories of resume in the target column. Out of 24 categories, information-technology and business-development classes are the largest with 119 resumes, whereas bpo class is the smallest with 21 resumes. I have encoded each category with integers from 0 to 23 using map function

My goal is to develop a machine learning classifier which is going to correctly predict the class of an applicant’s resume. Since I have 24 different categories, I will develop a multi-class classification algorithm

### 5. NLP steps to convert resume content to simplified text

Data cleaning must be done before vectorizing the text data. The steps I followed for cleaning the text data are,

1. Removing punctuations and non-ascii characters other that alphabets and numerics
2. Removing single and 2 letter words
3. Converting all the uppercase to lowercase texts
4. stop words removal
5. lemmatization to break a word down to its root

### 6. Fit XGBoost model on vectorized data

In NLP problems, we need to convert text data to numbers before applying any machine learning. That process is called the vectorization of the text data. Here vectorization has been done with Bag of Words (BoW) model

I used 1-gram TF-IDF approach and set maximum features limit to 3000. After vectorizing the data, I split the data into train and test data and fit the Extreme Gradient Boosting model to it

The model has been evaluated using the metric AUROC score. The AUROC score arrived is 0.96 which indicates that the model is of high accuracy and a good model

Refer resume_classifier.ipynb for the code block of the above steps

## Further Scope

The current project has successfully built and evaluated a machine learning model to predict the category of resume it belongs to. However there is still room for improvement and further scope in this project which includes,

1. Imbalanced data: this dataset is not balanced since each class is not represented equally well. The data need to be applied with balancing techniques in order to get more accurate results

2. Hyperparameter tuning: Basic xgboost model has been implemented in this project. The ML model requires further hypertuning to improvise it

3. Model Comparison: In addition to the model evaluated in this project, other classification models could also be implemented and compared to identify the best performing model for this problem

4. Exploratory Data Analysis (EDA): In detail EDA is further required to the data in order to further understand and train the models to the data

5. Deployment: The project need to be deployed as a proper app in some open-source app framework like streamlit 

6. NLP: Very basic NLP techniques has been applied in this project. Further deep techniques need to be applied in order to reduce the size of the data further down



