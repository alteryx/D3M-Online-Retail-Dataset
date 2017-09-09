import featuretools as ft
import featuretools.primitives.api as ftypes
import pandas as pd
from featuretools.variable_types import *
import sklearn.model_selection
import os
import argparse
import json


# Perfrom train test split, apply data section to both train and test
def dfs_d3m(inpath, outpath):
    # path to csv with labels, time, and customer id
    LABELS_PATH = "input/data/purchase_sum_4_weeks.csv"

    if not os.path.exists(outpath):
        try:
            os.makedirs(outpath)
            os.makedirs(os.path.join(outpath, 'data'))
        except OSError:
            print "Cannot make specified folder path.  Consider creating folders manually"
            pass

    # Create entityset from d3m multitable data
    es = multitable_d3m_to_entityset(inpath)

    # Load cutoff times
    cutoff_time = pd.DataFrame.from_csv(LABELS_PATH)
    cutoff_time['d3mIndex'] = cutoff_time.index.values
    cutoff_times_to_pass = cutoff_time[['time']]
    cutoff_times_to_pass['CustomerID'] = cutoff_times_to_pass.index.values

    # aggregation primitives to use
    agg_primitives = [ftypes.Sum, ftypes.Std, ftypes.Max, ftypes.Skew,
                      ftypes.Min, ftypes.Mean, ftypes.Count,
                      ftypes.PercentTrue, ftypes.NUnique, ftypes.Mode]

    # Calculate feature matrix
    fm, features = ft.dfs(entityset=es, target_entity='customers',
                          cutoff_time=cutoff_times_to_pass,
                          agg_primitives=agg_primitives, verbose=True)
    fm['d3mIndex'] = fm.index.values

    # Split data into train and test
    y = pd.read_csv(LABELS_PATH, index_col='CustomerID')['label']
    y.index.name = 'd3mIndex'
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(fm, y)

    # variables to pull from json
    name = 'online retail'
    redacted = "false"
    description_path = "../problemDescription.txt"
    citation_path = "../citation.bib"
    source = "ICS UCI Machine Learning Repository"
    remote_uri = "http://archive.ics.uci.edu/ml/datasets/online+retail"
    target_entity = 'customers'

    # Initialize the string with some basic metadata
    s = "{\n"
    s += "    \"dataSchema\": {\n"
    s += "        \"datasetId\": \"%s\",\n" % (es.id)
    s += "        \"redacted\": %s,\n" % (redacted)
    s += "        \"name\": \"%s\",\n" % (name)
    s += "        \"descriptionFile\": \"%s\",\n" % (description_path)
    s += "        \"citationFile\": \"%s\",\n" % (citation_path)
    s += "        \"source\": \"%s\",\n" % (source)
    s += "        \"remoteURI\": \"%s\",\n" % (remote_uri)
    s += "        \"rawData\": false,\n"
    s += "        \"dataSchemaVersion\": \"2.07\",\n"

    # TRAIN DATA

    s += "        \"%s\": {\n" % ("trainData")
    s += "            \"numSamples\": \"%d\",\n" % (X_train.shape[0])
    s += "            \"%s\":  [\n" % ("trainData")
    s += get_feature_types(X_train, features)
    s += "            ],\n"

    X_train.to_csv(os.path.join(outpath, 'data/trainData.csv'), index=False)

    # TRAIN TARGETS

    target_values = {
        'label': {'description': '', 'vartype': 'boolean', 'role': 'target'},
        'time': {'description': '', 'vartype': 'dateTime', 'role': 'metadata'},
        'd3mIndex': {'description': '', 'vartype': 'float', 'role': 'index'}
    }

    s += "            \"trainTargets\": [\n"

    for column in cutoff_time.columns:
        vals = target_values[column]
        s += "                {\n"
        s += "                    \"varName\": \"%s\",\n" % (column)
        s += "                    \"varDescription\": \"%s\",\n" % (vals['description'])
        s += "                    \"varType\": \"%s\",\n" % (vals['vartype'])
        s += "                    \"varRole\": \"%s\"\n" % (vals['role'])

        if column != cutoff_time.columns[-1]:
            s += "                },\n"
        else:
            s += "                }\n"

    s += "            ]\n"
    s += "        },\n"

    train_target_out = os.path.join(outpath, 'data/trainTargets.csv')
    cutoff_time.loc[y_train.index.values].to_csv(train_target_out, index=False)

    # TEST
    s += "        \"testDataSchemaMirrorsTrainDataSchema\": true,\n"
    s += "        \"%s\": {\n" % ("testData")
    s += "            \"numSamples\": \"%d\"\n" % (X_test.shape[0])
    s += "        }\n"
    s += "    }\n"
    s += "}"
    X_test.to_csv(os.path.join(outpath, 'data/testData.csv'), index=False)

    test_target_out = os.path.join(outpath, 'data/testTargets.csv')
    cutoff_time.loc[y_test.index.values].to_csv(test_target_out, index=False)

    # Write the string to file
    with open(os.path.join(outpath, 'dataSchema.json'), 'wb') as g:
        g.write(s)


def get_feature_types(feature_matrix, features):
    s = ""
    # Iterate over the features in the feature_matrix, storing their name/type/role
    feature_names = {f.get_name(): f for f in features}
    vtype_mappings = {Boolean: 'boolean', Categorical: 'categorical', Ordinal: 'ordinal', Text: 'text', Datetime: 'dateTime', Id: 'index'}
    for column in feature_matrix.columns:
        series = feature_matrix[column]
        if column in feature_names:
            feature = feature_names[column]
            role = 'attribute'
        elif column == 'd3mIndex':
            role = 'index'
        else:
            import pdb; pdb.set_trace()
        if feature.variable_type != Numeric:
            vartype = vtype_mappings[feature.variable_type]
        elif series.dtype in ['int16', 'int32', 'int64',]:
            vartype = 'integer'
        elif series.dtype in ['float16', 'float32', 'float64',]:
            vartype = 'float'

        s += "                {\n"
        s += "                    \"varName\": \"%s\",\n" % (column)
        s += "                    \"varType\": \"%s\",\n" % (vartype)
        s += "                    \"varRole\": \"%s\"\n" % (role)

        if column != feature_matrix.columns[-1]:
            s += "                },\n"
        else:
            s += "                }\n"

    return s


def multitable_d3m_to_entityset(inpath):
    with open(os.path.join(inpath, 'data/dataSchema.json'), 'rb') as f:
        raw_json = json.load(f)

    es = ft.EntitySet(raw_json['dataSchema']['datasetId'])

    relationships = []
    entities = [key[8:] for key in raw_json['dataSchema']
                if key.startswith("rawData/")]

    d3m_var_to_ft_var = {'boolean': Boolean, 'float': Numeric,
                         'zeroToOneFloat': Numeric, 'integer': Numeric,
                         'text': Text, 'categorical': Categorical,
                         'ordinal': Ordinal, 'dateTime': Datetime}

    for entity in entities:
        index = None
        csv_path = os.path.join(inpath, 'data/raw_data/%s.csv' % (entity))
        entityData = raw_json['dataSchema']['rawData/%s' % (entity)]['rawData/%s' % (entity)]
        var_types = {}
        for varData in entityData:
            if varData['varRole'] == 'index':
                index = varData['varName']
                var_types[varData['varName']] = Index
            elif 'varReference' in varData:
                var_types[varData['varName']] = Id
                parent_entity = varData['varReference']['references'][8:]
                parent_var_id = varData['varReference']['reference_id']
                relationships.append((parent_entity, parent_var_id, entity, varData['varName']))
            else:
                var_types[varData['varName']] = d3m_var_to_ft_var[varData['varType']]

        es.entity_from_csv(entity, csv_path, index=index, variable_types=var_types)

    for parent_entity, par_var_id, child_entity, child_var_id in relationships:
        relationship = ft.Relationship(es[parent_entity][par_var_id],
                                       es[child_entity][child_var_id])
        es.add_relationship(relationship)

    return es


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath')
    parser.add_argument('outpath')
    args = parser.parse_args()
    dfs_d3m(args.inpath, args.outpath)
