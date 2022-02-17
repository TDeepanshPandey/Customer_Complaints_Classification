from datetime import datetime
from pathlib import Path
from spacy.util import minibatch, compounding
import spacy
import pandas as pd
from spacy.training import Example
import json


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
    lab = list(y.columns)
    y = y.to_dict("index")
    dataset = list(zip(data[text_column], [{'cats': cats} for cats in y.values()]))
    return dataset, lab


def train_spacy(data, labels, output_path, iterations, validation_data, results_path):
    """
    input
        data: data in specified spacy format
        labels: a list containing all the labels for classification
        output_path: path to save the trained model
        iterations: number of epochs to train the model
        validation_data: validation data to check model after every epoch
        results_path: path to the folder where the validation results for every epoch will be saved in json file

    return
        nlp: trained model
    """
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat_multilabel")

    for i in labels:
        textcat.add_label(i)

    if validation_data:
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
            if validation_data:
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
