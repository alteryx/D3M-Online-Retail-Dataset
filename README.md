# About
The python script ``dfs_d3m.py`` takes in a multitable dataset
and outputs a feature matrix and D3M data schema.

# How to use
1. Install [python 2.7](https://www.python.org/downloads/release/python-2713/)
2. Install python requirements [Featuretools](https://www.featuretools.com/) and [scikit-learn](http://scikit-learn.org/stable/)

```
   pip install sklearn
   pip install featuretools
```

3. (Optional) Replace the LABELS_PATH in dfs_d3m.py to point to "data/purchase_sum_4_weeks_first_100.csv" to speed up run time (creates smaller dataset)
4. Run

```
python dfs_d3m.py input OUTPUT_PATH
```
(OUTPUT_PATH is the directory where the clean dataset will be created)

The outputs are:
* data/dataSchema.json
* data/trainData.csv
* data/trainTargets.csv
* data/testData.csv
* data/testTargets.csv


# Coming soon
1. This sample is currently hard coded to deal with the
   online retail data set. In the future, it will be able to
   take in an arbitrary data set.
2. Similarly, the code currently splits the data into test
   and train (because the online retail is all one set). It
   will in the future take in separate test data and also
   transform that.

# Reference
## General Links
* [Dataset Source](http://archive.ics.uci.edu/ml/datasets/online+retail)

## Wiki Links
* [Dataset Directory Structure](https://datadrivendiscovery.org/wiki/display/gov/Dataset+Directory+Structure)
* [Problem Annotation Schema](https://datadrivendiscovery.org/wiki/display/gov/Problem+Annotation+Schema)
* [Data Annotation Schema](https://datadrivendiscovery.org/wiki/display/gov/Data+Annotation+Schema)
