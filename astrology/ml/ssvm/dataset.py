import pandas as pd
import pkg_resources
from scipy.io import arff

def get_water_quality_dataset():
    # First we get data that demonstrates how
    # structural modeling work
    #
    # Thanks to https://discuss.analyticsvidhya.com/t/loading-arff-type-files-in-python/27419/2
    # on how to load this datatype

    fp = pkg_resources.resource_filename('ebmpy.datasets', "water-quality-nom.arff")
    data = arff.loadarff(fp)

    # Convert to a pandas dataframe
    df = pd.DataFrame(data[0])

    # Relabel labels
    label_map = {'25400': 'y1',
                 '29600': 'y2',
                 '30400': 'y3',
                 '33400': 'y4',
                 '17300': 'y5',
                 '19400': 'y6',
                 '34500': 'y7',
                 '38100': 'y8',
                 '49700': 'y9',
                 '50390': 'y10',
                 '55800': 'y11',
                 '57500': 'y12',
                 '59300': 'y13',
                 '37880': 'y14'}

    # Rename columns
    df = df.rename(columns=label_map)

    # Label columns
    label_cols = label_map.values()

    # Feature columns
    feature_cols = list(set(df.columns).difference(label_cols))

    # Cast each label column to an integer
    df = df.astype({l: 'int32' for l in label_cols})

    # Feature columns
    feature_cols = list(set(df.columns).difference(label_cols))

    return df, feature_cols, label_cols