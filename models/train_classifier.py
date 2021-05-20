import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    """Load data from database
    
    Args:
        Database file path
        
    Returns:
        messages array (n, ), category array (n, 40), category names (36, )
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Dataset', engine)
    X =  df.message.values
    Y =  df.iloc[:, 4:]
    print('shape of dataset is ', df.shape)
    
    return X, Y.values, Y.columns
    
def tokenize(text):
    """Tokenize text
    
    Change the text to lower case
    tokenize the word and remove stopwords
    lemmatize the tokens
    
    Args:
        text messages
        
    Returns:
        cleaned tokens
    """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Build a Machine Learning model
    
    Use pipeline to vectorize the text messages and tranform to tf-idf
    Apply multioutput classification using AdaBoostClassifier
    
    Args:
        None
        
    Returns:
        ML model
    
    """
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the prediction of the model
    
    Predict the test results from the model
    Report the f1 score, precision and recall for each output category of the dataset
    
    Args:
        model is the ML model created in build_model()
        X_test, Y_test are the tests data
        category_names is all the categories column names
    
    Returns:
        None
        Print the classification report for each column
    
    """
    
    Y_pred=model.predict(X_test)
    
    for i, label in enumerate(category_names):
        print(label)
        print(classification_report(Y_pred[:,i], Y_test[:,i]))
    

def save_model(model, model_filepath):
    """Save model to a pickle file
    
    Args:
        model is the ML model built before
        model_filepath is the pickle file path 
    
    Returns:
        None
        
    """   
    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
