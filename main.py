# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
import json
from spacy.training import Example
from pathlib import Path


def prepare_data():
    """
    Function to change complaints data csv to use case (text and label) dataframe
    :return: Save a pickle
    :rtype: None
    """
    df = pd.read_csv('data/complaints.csv', nrows=200000)
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
    df = df[['Consumer complaint narrative', 'category_id']]
    pd.to_pickle(df, 'data/dataset.pickle')


def prepare_spacy_dataset(data, text_column, label_column):
    """
    Function to prepare dataset into a format for spacy
    :param data: dataframe containing the dataset
    :type data: dataframe
    :param text_column: column with the document or text data
    :type text_column: string
    :param label_column: column with labels or where multi classification needs to be done
    :type label_column: string
    :return: list containing text/documents and one hot encoded labels
    :rtype: list
    """
    y = pd.get_dummies(data[label_column], prefix='label')
    labels = list(y.columns)
    y = y.to_dict("index")
    dataset = list(zip(data[text_column], [{'cats': cats} for cats in y.values()]))
    return dataset, labels


def train_spacy(data, labels, output_path, iterations, validation_data, results_path):
    """
    input
        data: data in specified spacy format
        labels: a list containing all the labels for classification
        save_flag: boolean True for saving model otherwise False
        output_path: path to save the trained model
        iterations: number of epochs to train the model
        config: spacy initialize configuration
        validation_data: validation data to check model after every epoch
        results_path: path to the folder where the validation results for every epoch will be saved in json file

    return
        nlp: trained model
    """
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat_multilabel", config=config)

    for i in labels:
        textcat.add_label(i)

    if not validation_data:
        val_dataset = []
        for val_texts, val_annotations in validation_data:
            doc = nlp.make_doc(val_texts)
            val_dataset.append(Example.from_dict(doc, val_annotations))

    optimizer = nlp.begin_training()
    start_time = datetime.now()
    print("Training the model...", )

    with nlp.select_pipes(enable="textcat_multilabel"):  # only train textcat
        for j in range(iterations):
            iter_start_time = datetime.now()
            losses = {}
            k = 0
            # batch up the examples using spaCy's minibatch
            batches = minibatch(data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                example = []
                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    example.append(Example.from_dict(doc, annotations[i]))
                nlp.update(example, sgd=optimizer, drop=0.2, losses=losses)
                print('Batch No.: {} Loss - {}'.format(k, round(losses['textcat_multilabel'], 3)))
                k += 1
            print("\n\nCompleted Iterations : {}".format(j))
            print("Total Iteration Time :", datetime.now() - iter_start_time)
            if not validation_data:
                score = nlp.evaluate(val_dataset)
                print("Validation Results :")
                print(json.dumps(score['cats_auc_per_type'], indent=4, sort_keys=True))
                if results_path != '':
                    with open(results_path+'validation_results_iteration_'+str(j)+'.json', 'w', encoding='utf-8') as f:
                        json.dump(score['cats_auc_per_type'], f, ensure_ascii=False, indent=4)
            print("\n\n")
    print("Total Training Time - ", datetime.now() - start_time)
    if output_path is not None:
        output_dir = Path(output_path)
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    return nlp


if __name__ == '__main__':
    #prepare_data()
    df = pd.read_pickle('data/dataset.pickle')
    data, labels = prepare_spacy_dataset(df, 'Consumer complaint narrative', 'category_id')
    train_data, test_dataset = train_test_split(data, train_size=0.8, random_state=42)
    test, val_data = train_test_split(test_dataset, test_size=0.25, random_state=42)
    nlp = train_spacy(train_data, labels, save_flag, output_path, iterations, config, val_data, results_path)
    test_data = []
    for texts, annotations in test:
        doc = nlp.make_doc(texts)
        test_data.append(Example.from_dict(doc, annotations))

    score = nlp.evaluate(test_data)
    print(json.dumps(score['cats_auc_per_type'], indent=4, sort_keys=True))
    #print(data[0])
    #print(labels)

