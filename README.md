Telco-Customer-Churn-Prediction-Using-Machine-Learning-Models
building a model that accurately predicts customer churnIntro
General
Machine learning allows the user to feed a computer algorithm an immense amount of data and have the computer analyze and make data-driven recommendations and decisions based on only the input data. In most of the situations we want to have a machine learning system to make predictions, so we have several categories of machine learning tasks depending on the type of prediction needed: Classification, Regression, Clustering, Generation, etc.
Classification is the task whose goal is the prediction of the label of the class to which the input belongs (e.g., Classification of images in two classes: cats and dogs). Regression is the task whose goal is the prediction of numerical value(s) related to the input (e.g., House rent prediction, Estimated time of arrival). Generation is the task whose goal is the creation of something new related to the input (e.g., Text translation, Audio beat generation, Image denoising). Clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other clusters (e.g., Client’s clustering).
In machine learning, there are learning paradigms that relate to one aspect of the dataset: the presence of the label to be predicted. Supervised Learning is the paradigm of learning that is applied when the dataset has the label variables to be predicted, known as y variables. Unsupervised Learning is the paradigm of learning that is applied when the dataset has not the label variables to be predicted. Self-supervised Learning is the paradigm of learning that is applied when part of the X dataset is considered as the label to be predicted (e.g., the Dataset is made of texts and the model try to predict the next word of each sentence).
**Notebook overview**
This notebook is a guide to start practicing Machine Learning.
Project Title: Telco Customer Churn Prediction Using Machine Learning Models
Project Description: To build a model that accurately predicts customer churn
Hypothesis
Null Hypothesis: Customers churn based on Online Security, Tech Support and Total Charges
Alternate Hypothesis: Customers do not churn based on Online Security, Tech Support and Total Charges
Questions
1. Do customers churn based on online security?
2. Do customers churn based on tech support?
3. Do customers churn due to total charges?
4. Do customers churn due to monthly charges?
5. Does the payment method contribute to why customers churn?
6. Does the billing method contribute to why customers churn?
7. Are customers churning based on the contract type?
**Setup**
**Installation**
Here is the section to install all the packages/libraries that will be needed to tackle the challenge.
 
**Importation**
Here is the section to import all the packages/libraries that will be used through this notebook.
 
**Data Loading**
Here is the section to load the datasets (train, eval, test) and the additional files
 
After loading data the previous steps is to check the shape of data, checking duplication, checking data information, checking null values in each columns, converting data types, creating new data frame containing only categorical data, calling function to return unique values, changing value from 0 and 1 to no and yes, and so on.
**Data Understanding**
The data for this project is in a csv format. The following describes the columns present in the data.
Gender: Whether the customer is a male or a female
Senior Citizen: Whether a customer is a senior citizen or not
Partner: Whether the customer has a partner or not (Yes, No)
Dependents: Whether the customer has dependents or not (Yes, No)
Tenure: Number of months the customer has stayed with the company
Phone Service: Whether the customer has a phone service or not (Yes, No)
Multiple Lines: Whether the customer has multiple lines or not
Internet Service: Customer's internet service provider (DSL, Fiber Optic, No)
Online Security: Whether the customer has online security or not (Yes, No, No Internet)
Online Backup: Whether the customer has online backup or not (Yes, No, No Internet)
Device Protection: Whether the customer has device protection or not (Yes, No, No internet service)
Tech Support: Whether the customer has tech support or not (Yes, No, No internet)
Streaming TV: Whether the customer has streaming TV or not (Yes, No, No internet service)
Streaming Movies: Whether the customer has streaming movies or not (Yes, No, No Internet service)
Contract: The contract term of the customer (Month-to-Month, One year, Two year)
Paperless Billing: Whether the customer has paperless billing or not (Yes, No)
Payment Method: The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))
Monthly Charges: The amount charged to the customer monthly
Total Charges: The total amount charged to the customer
Churn: Whether the customer churned or not (Yes or No)
Exploratory Data Analysis: EDA
Here is the section to inspect the datasets in depth, present it, make hypotheses and think the cleaning, processing and features creation.
 
**Hypothesis Testing**
 
Univariate Analysis
Visualizing all categorical columns
 
In this section we tried to check each column and counting the total of unique values of gender column where the male gender dominated the use of telco services with a total of 3555 against 3488 of the female gender, senior citizen column where The total of Senior Citizens who use telco services are few with a total of 1142 compared to those who aren't senior citizens, partner column where Customers without partners who use telco services are more with a total of 3641 than those who have partners, dependents column where Few customers have dependents with a total of 2110 compared to those without dependents (4933), phone service column where Customers who use phone services are more with a total of 6361 compared to those who don't use phone services(682), and so on.
     
After all those methods I started to answer the asked questions
1.	Do customers churn based on online security?
•	With regards to Online Security, customers will either churn or not churn based on whether they have subscribed to the service or not. From the visualization, it can be seen that customers with no online security are not churning, and it's same for those with online security and those with no internet service.
•	We can say that customers do not churn based on Online Security services by the Telco Company.
                               
2.	Do customers churn based on tech support?
•	With regards to Tech Support, customers will either churn or not churn based on whether they have subscribed to the service or not. From the visualization, it can be seen that customers with no tech support are not churning, and it's same for those with online security and those with no internet service.
•	We can say that customers do not churn based on Tech Support services by the Telco Company

                                      
3.	Do customers churn due to total charges?
•	Customers with high total charges do not churn. This may be due to the services they're receiving.
 
4.	Do customers churn due to monthly charges?
•	Customers with high monthly charges do not churn. This may be due to the services they're receiving.
 
5.	Does the payment method contribute to why customers churn?
•	Customers are not likely to churn based on payment method
 
6.	Does the billing method contribute to why customers churn?
•	Customers are not churning based on billing system
 
7.	Are customers churning based on the contract type?
•	Customers are not churning based on contract type
 
Build & Select Model: Train Model on dataset and select the best performing model
Here is the section to build, train, evaluate and compare the models to each other’s where I tried to use different models which are decision tree, random forest, support vector machine, LightGBM, Kneighbors classifier, XBoost, Gradient Boosting, Stochastic Gradient Boosting, Logistic Regression, where I tried to create model, train model, evaluate a model on the evaluation dataset and predict on unknown dataset.
After those models I started to make a model’s comparison where I used a pandas data frame method. sort_values() to sort the data frame regarding the metric.
      
**Hyperparameters tuning**
Fine-tune the Top-k models (3 < k < 5) using a GridSearchCV (that is in sklearn.model_selection ) to find the best hyperparameters and achieve the maximum performance of each of the Top-k models, then compare them again to select the best one.

**Key Insights and Conclusion**.
From the EDA performed we realized that the data is imbalanced since the number of customers who aren't churning made up the majority of the dataset.
In dealing with this issue:
1. All other evaluation metrics will be used except 'accuracy' since it will only the majority class
2. The type of machine learning algorithms that will be used are the Tree based models. These algorithms work by learning a hierarchy of if/else questions. This can force both classes to be addressed.



