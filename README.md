# About
The python script ~/dfw_d3m/dfs_d3m.py~ takes in a multitable dataset
and outputs a feature matrix and D3M data schema.

# How to use
1. Install [[https://www.python.org/downloads/release/python-2713/][python 2.7]]
2. Install python requirements [[https://www.featuretools.com/][Featuretools]] and [[http://scikit-learn.org/stable/][scikit-learn]]

```
   pip install sklearn
   pip install featuretools
```

3. Download the tar file at https://s3.amazonaws.com/featuretools-static/d3m_dfs.tar.gz
4. Replace the LABELS_PATH and CUSTOMERS_PATH in dfs_d3m.py to point to customers.csv and purchase_sum_4_weeks.csv (or the _first_100 versions of those files)
5. Run python dfs_d3m.py input_path output_path

This takes about 30 minutes to generate a feature matrix
with the full dataset and about 30 seconds with the included
sample. The outputs are:
* dataSchema.json
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
** General Links
+ [[http://archive.ics.uci.edu/ml/datasets/online+retail][Dataset Source]]

## Wiki Links
* [[https://datadrivendiscovery.org/wiki/display/gov/Dataset+Directory+Structure][Dataset Directory Structure]]
* [[https://datadrivendiscovery.org/wiki/display/gov/Problem+Annotation+Schema][Problem Annotation Schema]]
* [[https://datadrivendiscovery.org/wiki/display/gov/Data+Annotation+Schema][Data Annotation Schema]]
