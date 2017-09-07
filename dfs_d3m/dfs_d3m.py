import featuretools as ft
import featuretools.primitives.api as ftypes
import pandas as pd
from featuretools.variable_types import *
import sklearn.model_selection
import copy
import os
import argparse


# Perfrom train test split, apply data section to both train and test
def dfs_d3m(inpath, outpath):
    # path to csv with labels, time, and customer id
    LABELS_PATH = "/home/rwedge/Online-Retail-Dataset/data/purchase_sum_4_weeks.csv"
    # path to csv with customers table
    CUSTOMERS_PATH = "/home/rwedge/Online-Retail-Dataset/data/customers.csv"

    if not os.path.exists(outpath):
        try:
            os.makedirs(outpath)
            os.makedirs(os.path.join(outpath, "data"))
        except OSError:
            print "Cannot make specified folder path.  Consider creating folders manually"
            pass

    y = pd.read_csv(LABELS_PATH, index_col="CustomerID")["label"]
    X = pd.read_csv(CUSTOMERS_PATH, index_col="CustomerID")
    X["CustomerID"] = X.index.values
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)

    # Load retail dataset and create train and test entitysets
    es = ft.demo.load_retail()
    train_es = copy.deepcopy(es)
    test_es = copy.deepcopy(es)

    train_es["customers"].update_data(X_train)
    test_es["customers"].update_data(X_test)

    # Load cutoff times
    cutoff_time = pd.DataFrame.from_csv(LABELS_PATH)
    cutoff_times_to_pass = cutoff_time[["time"]]
    cutoff_times_to_pass["CustomerID"] = cutoff_times_to_pass.index.values
    train_cutoff_time = cutoff_times_to_pass.loc[y_train.index.values]
    test_cutoff_time = cutoff_times_to_pass.loc[y_test.index.values]

    # variables to pull from json
    name = "online retail"
    redacted = "false"
    description_path = "../problemDescription.txt"
    citation_path = "../citation.bib"
    source = "ICS UCI Machine Learning Repository"
    remote_uri = "http://archive.ics.uci.edu/ml/datasets/online+retail"
    target_entity = "customers"

    # aggregation primitives to use
    agg_primitives = [ftypes.Sum, ftypes.Std, ftypes.Max, ftypes.Skew,
                      ftypes.Min, ftypes.Mean, ftypes.Count,
                      ftypes.PercentTrue, ftypes.NUnique, ftypes.Mode]

    # Initialize the string with some basic metadata
    s = "{\n"
    s += "    \"dataSchema\": {\n"
    s += "        \"datasetId\": \"%s\",\n" % (train_es.id)
    s += "        \"redacted\": %s,\n" % (redacted)
    s += "        \"name\": \"%s\",\n" % (name)
    s += "        \"descriptionFile\": \"%s\",\n" % (description_path)
    s += "        \"citationFile\": \"%s\",\n" % (citation_path)
    s += "        \"source\": \"%s\",\n" % (source)
    s += "        \"remoteURI\": \"%s\",\n" % (remote_uri)
    s += "        \"rawData\": false,\n"
    s += "        \"dataSchemaVersion\": \"2.07\",\n"

    # TRAIN DATA
    fm, features = ft.dfs(entityset=train_es, target_entity=target_entity,
                          cutoff_time=train_cutoff_time,
                          agg_primitives=agg_primitives, verbose=True)
    fm["d3mIndex"] = fm.index.values
    s += "        \"%s\": {\n" % ("trainData")
    s += "            \"numSamples\": \"%d\",\n" % (fm.shape[0])
    s += "            \"%s\":  [\n" % ("trainData")
    s += get_feature_types(fm, features)
    s += "            ],\n"

    fm.to_csv(os.path.join(outpath, "data/trainData.csv"), index=False)

    # TRAIN TARGETS
    cutoff_time["d3mIndex"] = cutoff_time.index.values

    target_values = {
        "label": {"description": "", "vartype": "boolean", "role": "target"},
        "time": {"description": "", "vartype": "dateTime", "role": "metadata"},
        "d3mIndex": {"description": "", "vartype": "float", "role": "index"}
    }

    s += "            \"trainTargets\": [\n"

    for column in cutoff_time.columns:
        vals = target_values[column]
        s += "                {\n"
        s += "                    \"varName\": \"%s\",\n" % (column)
        s += "                    \"varDescription\": \"%s\",\n" % (vals["description"])
        s += "                    \"varType\": \"%s\",\n" % (vals["vartype"])
        s += "                    \"varRole\": \"%s\",\n" % (vals["role"])
        s += "                },\n"

    s += "            ]\n"
    s += "        },\n"

    train_target_out = os.path.join(outpath, "data/trainTargets.csv")
    cutoff_time.loc[y_train.index.values].to_csv(train_target_out, index=False)

    # TEST
    s += "        \"testDataSchemaMirrorsTrainDataSchema\": true,\n"

    fm, features = ft.dfs(entityset=test_es, target_entity=target_entity,
                          cutoff_time=test_cutoff_time,
                          agg_primitives=agg_primitives, verbose=True)

    s += "        \"%s\": {\n" % ("testData")
    s += "            \"numSamples\": \"%d\",\n" % (fm.shape[0])
    s += "        },\n"
    s += "    }\n"
    s += "}"
    fm.to_csv(os.path.join(outpath, "data/testData.csv"), index=False)

    test_target_out = os.path.join(outpath, "data/testTargets.csv")
    cutoff_time.loc[y_test.index.values].to_csv(test_target_out, index=False)

    # Write the string to file
    with open(os.path.join(outpath, "dataSchema.json"), "wb") as g:
        g.write(s)


def get_feature_types(feature_matrix, features):
    s = ""
    # Iterate over the features in the feature_matrix, storing their name/type/role
    feature_names = {f.get_name(): f for f in features}
    vtype_mappings = {Boolean: "boolean", Categorical: "categorical", Ordinal: "ordinal", Text: "text", Datetime: "dateTime", Id: "index"}
    for column in feature_matrix.columns:
        series = feature_matrix[column]
        if column in feature_names:
            feature = feature_names[column]
            role = "attribute"
        elif column == "d3mIndex":
            role = "index"
        else:
            import pdb; pdb.set_trace()
        if feature.variable_type != Numeric:
            vartype = vtype_mappings[feature.variable_type]
        elif series.dtype in ["int16", "int32", "int64",]:
            vartype = "integer"
        elif series.dtype in ["float16", "float32", "float64",]:
            vartype = "float"

        s += "                {\n"
        s += "                    \"varName\": \"%s\",\n" % (column)
        s += "                    \"varType\": \"%s\",\n" % (vartype)
        s += "                    \"varRole\": \"%s\",\n" % (role)
        s += "                },\n"
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath")
    parser.add_argument("outpath")
    args = parser.parse_args()
    dfs_d3m(args.inpath, args.outpath)
