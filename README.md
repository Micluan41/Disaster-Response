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

**app**
- **template**
    - master.html # main page of web app
    - go.html # classification result page of web app
- run.py # Flask file that runs app

**data**
- disaster_categories.csv # data to process
- disaster_messages.csv # data to process
- process_data.py
- InsertDatabaseName.db # database to save clean data to

**models**
- train_classifier.py # python script to build model with AdaBoostClassifier
- train_classifier_knn.py # python script to build model with KNeighborsClassifier
- classifier.pkl # saved model
- knn_classifier.pkl # saved model

README.md

## **Code Execution**
1. Run 'process_data.py' to clean the data and save to a database
   - *python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db*
   
2. Run 'train_classifier.py' or 'train_classifier_cnn.py' to build the model and save as a pickle file
   - *python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl* **or**
   - *python models/train_classifier_knn.py data/DisasterResponse.db models/knn_classifier.pkl*

3. Run 'run.py' in webapp folder to launch the web app. Then, open the [web address](https://view6914b2f4-3001.udacity-student-workspaces.com/) in a browser.

#### **Note**
When running 'process_data.py' and 'train_classifier.py', follow the comments in the main function to provide the file path.


## **Additional Note**
'train_classifier.py' uses AdaBoostClassifier in the pipeline but does not have a GridSearch and runs fast. 
'train_classifier_knn.py' has GridSearch and therefore could run for a long time. 
In 'run.py', the code will load the pkl to fetch the model, the default is for 'train_classifier.py'. Change the model name in 'run.py' if you want to use the knn classifier in the web app.


