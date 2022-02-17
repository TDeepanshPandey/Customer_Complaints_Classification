# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from spacy.training import Example
import matplotlib.pyplot as plt
from utils.spacy_utils import prepare_spacy_dataset, train_spacy

# Setting the Parameters
output_path = 'models/'
iterations = 3
results_path = 'results/'


def prepare_data():
    """
    Function to change complaints data csv to use case (text and label) dataframe
    :return: Save a pickle
    :rtype: None
    """
    df = pd.read_csv('data/complaints.csv', nrows=20000)
    df = df[['Consumer complaint narrative', 'Product']]
    df = df.dropna(axis=0)
    print("Size of dataset", len(df), 'x', len(df.columns))
    count_val = df.groupby("Product")["Product"].count().sort_values(ascending=False)
    count_val = count_val[:10]
    top_10_val = [i in count_val for i in df.Product]
    df['label'] = top_10_val
    df.loc[df['label'] == False, 'Product'] = 'Others'
    count_val = df.groupby("Product")["Product"].count().sort_values(ascending=False)
    df['category_id'] = df['Product'].factorize()[0]
    df = df[['Consumer complaint narrative', 'Product', 'category_id']]
    df.columns = ['Consumer_complaint_narrative', 'Product', 'category_id']
    pd.to_pickle(df, 'data/dataset.pickle')


def plot_data(df):
    """
    Function to plot category ids in histogram
    :param df: dataframe containing all the data and column for histogram
    :type df: dataframe
    """
    fig = plt.figure(figsize=(16, 20))
    plt.hist(df.Product, color='blue',rwidth=0.8)
    plt.xticks(rotation=90)
    plt.xlabel("Category ID")
    plt.ylabel("Number of Complaints")
    plt.title("Number of Complaints for every category")
    plt.savefig('data/complaints_histogram.png')
    plt.show()


if __name__ == '__main__':
    prepare_data()
    df = pd.read_pickle('data/dataset.pickle')
    plot_data(df)
    data, labels = prepare_spacy_dataset(df, 'Consumer_complaint_narrative', 'category_id')
    train_data, test_dataset = train_test_split(data, train_size=0.8, random_state=13)
    test, val_data = train_test_split(test_dataset, test_size=0.25, random_state=13)

    nlp = train_spacy(train_data, labels, output_path, iterations, val_data, results_path)

    test_data = []
    for texts, annotations in test:
        doc = nlp.make_doc(texts)
        test_data.append(Example.from_dict(doc, annotations))

    score = nlp.evaluate(test_data)
    print(json.dumps(score['cats_auc_per_type'], indent=4, sort_keys=True))
    with open(results_path + 'testset_results.json', 'w', encoding='utf-8') as f:
        json.dump(score['cats_auc_per_type'], f, ensure_ascii=False, indent=4)



