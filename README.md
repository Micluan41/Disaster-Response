## **Table of Contents**
1. Installation
2. Project Summary
3. File Description
4. Code Execution
5. Additional Note

## **Installation**
Some NLP packages are needed to run the python script, such as **nltk, re**. **sqlalchemy** is used to load the data to a database. **sklearn** is required to build the machine learning model and **pickle** is used to save the model. **flask, plotly,** and **json** are imported to show data visualization in the web app. 

## **Project Summary**
In this project, I used a data set containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that the messages could be sent to an appropriate disaster relief agency.

The project also included a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

## **File Description**
Data sets 'disaster_messages.csv' and 'disaster_categories.csv' are in the **Data** folder

### **Python Script**
'process_data.py' cleans the raw data and create a database
'train_classifier.py' and 'train_classifier_knn.py' build machine learning pipeline using AdaBoost and KNN classifiers respectively.

in the **webapp** folder, 'go.html' and 'master.html' formats the web app and 'run.py' creates the data visualization and the results of input messages from the model. 

## **Code Execution**
1. Run 'process_data.py' to clean the data and save to a database
2. Run 'train_classifier.py' or 'train_classifier_cnn.py' to build the model and save as a pickle file
3. Run 'run.py' to launch the web app. Then, open the [web address](https://view6914b2f4-3001.udacity-student-workspaces.com/) in a browser

## **Additional Note**
'train_classifier.py' uses AdaBoostClassifier in the pipeline but does not have a GridSearch and runs fast. 
'train_classifier_knn.py' has GridSearch and therefore could run for a long time. 


