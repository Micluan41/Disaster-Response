import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Load data file
    
    Load messages and categories data and merge to a dataframe
    
    Args:
        messages and categories data file path
        
    Return:
        Dataframe df
    """
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    
    return df


def clean_data(df):
    """Clean loaded data for ML model
    
    Split 'categories' into separate category columns and rename with new column names
    Convert category values to just numbers 0 or 1 
    Drop the original categories column of df and replace it with new category columns
    Remove duplicates ids 
    Replace class 2 of 'related' category to 1
    
    Args:
        Raw dataframe
        
    Returns:
        Dataframe after cleaning
    """
    categories = df['categories'].str.split(';', expand=True)
    
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x.split('-')[0]).values
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]    
        categories[column] = pd.to_numeric(categories[column])

    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], join='inner', axis=1)
    
    df=df.drop_duplicates(subset=['id'])

    df['related'].replace(2, 1, inplace=True)
    
    return df


def save_data(df, database_filename):
    """Save the clean dataset into an sqlite database
    
    Args:
        Dataframe df
        database file name 
    
    Returns:
        None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Dataset', engine, if_exists='replace', index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        print('Shape of the dataframe', df.shape)
        print('uique id in df', df['id'].nunique())
 
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
