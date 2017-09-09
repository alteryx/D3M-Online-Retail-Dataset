import os
from featuretools.variable_types import *


def entityset_to_d3m(entityset, outpath, source=None, name=None,
                     remote_uri=None, description_path=None, citation_path=None,
                     redacted=True):
    # use schema defaults
    assert type(redacted) == bool, 'redacted must be a boolean'
    redacted = str(redacted).lower()
    description_path = description_path or "../problemDescription.txt"
    citation_path = citation_path or "../citation.bib"

    # check if outpath exists
    if not os.path.exists(outpath):
        try:
            os.makedirs(outpath)
            os.makedirs(os.path.join(outpath, 'data/raw_data'))
        except OSError:
            print "Cannot make specified folder path.  Consider creating folders manually"
            pass

    if len(entityset.entities) > 1:
        raw_data = 'true'

    # Initialize the string with some basic metadata
    s = "{\n"
    s += "    \"dataSchema\": {\n"
    s += "        \"datasetId\": \"%s\",\n" % (entityset.id)
    s += "        \"redacted\": %s,\n" % (redacted)
    if name is not None:
        s += "        \"name\": \"%s\",\n" % (name)
    s += "        \"descriptionFile\": \"%s\",\n" % (description_path)
    s += "        \"citationFile\": \"%s\",\n" % (citation_path)
    if source is not None:
        s += "        \"source\": \"%s\",\n" % (source)
    if remote_uri is not None:
        s += "        \"remoteURI\": \"%s\",\n" % (remote_uri)
    s += "        \"rawData\": %s,\n" % (raw_data)
    s += "        \"dataSchemaVersion\": \"2.07\",\n"

    vtype_mappings = {Boolean: 'boolean', Categorical: 'categorical', Ordinal: 'ordinal', Text: 'text', Datetime: 'dateTime', DatetimeTimeIndex: 'dateTime', Id: 'categorical', Index: 'categorical'}

    references = {}
    for r in entityset.relationships:
        references[(r.child_entity.id, r.child_variable.id)] = (r.parent_entity.id, r.parent_variable.id)

    for entity in entityset.entities:
        entity_path = os.path.join(outpath, 'data/raw_data/%s.csv' % (entity.id))
        entity.df.to_csv(entity_path, index=False)
        s += "        \"rawData/%s\": {\n" % (entity.id)
        s += "            \"numSamples\": \"%d\",\n" % (entity.df.shape[0])
        s += "            \"rawData/%s\": [\n" % (entity.id)

        for variable in entity.variables:
            series = entity.df[variable.id]
            if type(variable) != Numeric:
                vartype = vtype_mappings[type(variable)]
            elif series.dtype in ['int16', 'int32', 'int64',]:
                vartype = 'integer'
            elif series.dtype in ['float16', 'float32', 'float64',]:
                vartype = 'float'

            s += "                {\n"
            s += "                    \"varName\": \"%s\",\n" % (variable.name)
            s += "                    \"varType\": \"%s\",\n" % (vartype)

            if type(variable) == Index:
                role = 'index'
            else:
                role = 'attribute'

            if (entity.id, variable.id) in references:
                parent_id, p_var_id = references[(entity.id, variable.id)]
                s += "                    \"varRole\": \"%s\",\n" % (role)
                s += "                    \"varReference\": {\n"
                s += "                        \"references\": \"rawData/%s\",\n" % (parent_id)
                s += "                        \"reference_id\": \"%s\"\n" % (p_var_id)
                s += "                     }\n"
            else:
                s += "                    \"varRole\": \"%s\"\n" % (role)

            if variable != entity.variables[-1]:
                s += "                },\n"
            else:
                s += "                }\n"

        s += "            ]\n"

        if entity != entityset.entities[-1]:
            s += "        },\n"
        else:
            s += "        }\n"

    # Link var ->  File, Link var
    # Add trainData / trainTargets variable data to schema

    s += "    }\n"
    s += "}"

    # write json to file
    with open(os.path.join(outpath, 'data/dataSchema.json'), 'wb') as f:
        f.write(s)
