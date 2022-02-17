# Customer Complaints Multiclass Text Classification
In this repository we do multi class text classification on customer complaints data with Spacy library.

The data is available at - https://catalog.data.gov/dataset/consumer-complaint-database 

**Note** - Due to computational power limitation. I have used only 20000 rows of the dataset.

The distribution of data after removing NA values is shown as below - 
![alt text](https://github.com/TDeepanshPandey/Customer_Complaints_Classification/blob/master/data/complaints_histogram.png)

**Results (Area Under Curver for every Category)** - 
```
{
    "label_0": 0.9401665082245845,
    "label_1": 0.9143256286512573,
    "label_2": 0.9788608641720073,
    "label_3": 0.925170708735578,
    "label_4": 0.6409774436090226,
    "label_5": 0.8382632293080055,
    "label_6": 0.986039886039886,
    "label_7": 0.7642045454545454,
    "label_8": 0.9304035567715458,
    "label_9": null,
    "label_10": 0.23288590604026843
}
```
