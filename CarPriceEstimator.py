import platform;
import pandas as pd
import numpy as np
import tensorflow as tf

# Verify the versions of the included packages
print('Panda Version = ', pd.__version__)
print('Numpy Version = ', np.__version__)
print('Tensorflow Version = ', tf.__version__)

print(platform.architecture())

# The CSV file does not have a header, so we have to fill in column names.
names = [
    'symboling',
    'normalized-losses',
    'make',
    'fuel-type',
    'aspiration',
    'num-of-doors',
    'body-style',
    'drive-wheels',
    'engine-location',
    'wheel-base',
    'length',
    'width',
    'height',
    'curb-weight',
    'engine-type',
    'num-of-cylinders',
    'engine-size',
    'fuel-system',
    'bore',
    'stroke',
    'compression-ratio',
    'horsepower',
    'peak-rpm',
    'city-mpg',
    'highway-mpg',
    'price',
]

# We also have to specify dtypes.
dtypes = {
    'symboling': np.int32,
    'normalized-losses': np.float32,
    'make': str,
    'fuel-type': str,
    'aspiration': str,
    'num-of-doors': str,
    'body-style': str,
    'drive-wheels': str,
    'engine-location': str,
    'wheel-base': np.float32,
    'length': np.float32,
    'width': np.float32,
    'height': np.float32,
    'curb-weight': np.float32,
    'engine-type': str,
    'num-of-cylinders': str,
    'engine-size': np.float32,
    'fuel-system': str,
    'bore': np.float32,
    'stroke': np.float32,
    'compression-ratio': np.float32,
    'horsepower': np.float32,
    'peak-rpm': np.float32,
    'city-mpg': np.float32,
    'highway-mpg': np.float32,
    'price': np.float32,
}

# Read the file
mydata = pd.read_csv('data\imports-85.data', names=names, dtype=dtypes, na_values='?')

# Drop data rows that has no price data
mydata = mydata.dropna(axis='rows', how='any', subset=['price'])

# Fill floating data columns with 0
numeric_columns = [k for k, v in dtypes.items() if v == np.float32]
mydata[numeric_columns] = mydata[numeric_columns].fillna(value=0., axis='columns')

# Fill string data columns with ''
string_columns = [k for k, v in dtypes.items() if v == np.str]
mydata[string_columns] = mydata[string_columns].fillna(value='', axis='columns')

# Split data into training and estimation
training_data = mydata[:160]
evaluation_data = mydata[160:]

# Separate input features from data
training_label = training_data.pop('price')
evaluation_label = evaluation_data.pop('price')


# Make input function (using pandas) for data
# Can use any generators like numpy_input_fn or generator_input_fn
training_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_data, y=training_label,
                                                        batch_size=64, shuffle=True,
                                                        num_epochs=None)

# Make input function for evaluation
# Can use any generators like numpy_input_fn or generator_input_fn
evaluation_input_fn = tf.estimator.inputs.pandas_input_fn(x=evaluation_data, y=evaluation_label,
                                                        batch_size=64, shuffle=True,
                                                        num_epochs=None)

# Define the three different types of inputs
make = tf.feature_column.categorical_column_with_hash_bucket('make', 100)
horsepower = tf.feature_column.numeric_column('horsepower', shape=[])
num_of_cylinders = tf.feature_column.categorical_column_with_vocabulary_list(
    'num-of-cylinders', ['two', 'three', 'four'])

# Apply linear regression machine learning algorithm
# Can use other models (DNNRegressor, DNNClassifier, TensorForest, KMeans)
regressor = tf.contrib.learn.LinearRegressor(feature_columns=[make, horsepower, num_of_cylinders])

# Train my thing
regressor.fit(input_fn=training_input_fn, steps=10000)

regressor.evaluate(input_fn=evaluation_input_fn)