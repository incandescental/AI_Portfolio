import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report, hamming_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import RandomizedSearchCV

from tensorflow import keras
from tensorflow_addons.metrics import HammingLoss
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import keras_tuner as kt

# Import data
# Originally from https://www.kaggle.com/purumalgi/music-genre-classification

df = pd.read_csv('https://raw.githubusercontent.com/incandescental/AI_Portfolio/main/train.csv')
labels = pd.read_csv('https://raw.githubusercontent.com/incandescental/AI_Portfolio/main/submission.csv')

# Assign ID Values

df = df.sort_values(['Artist Name', 'Track Name'])

df['TrackID'] = df.groupby(['Artist Name', 'Track Name']).ngroup()

# Remove duplicates

df = df.drop(columns = ['Artist Name', 'Track Name'])
df = df.drop_duplicates(subset = ['TrackID', 'Class'])

# Fill numeric missing values with average for class

df.isna().sum()

popularity_means = df.groupby('Class')['Popularity'].mean().tolist()
df.Popularity = [popularity_means[y] if np.isnan(x) else x for x,y in zip(df.Popularity, df.Class)]

instrumentalness_means = df.groupby('Class')['instrumentalness'].mean().tolist()
df.instrumentalness = [instrumentalness_means[y] if np.isnan(x) else x for x,y in zip(df.instrumentalness, df.Class)]

# Fill categorical variable with most common value for that class

key_count = df.groupby(['Class', 'key']).agg(Count = ('key', 'count')).reset_index()
key_count_max = key_count.loc[key_count.groupby('Class')['Count'].idxmax()]

df.key = [key_count_max.key.iloc[y] if np.isnan(x) else x for x,y in zip(df.key, df.Class)]

# Create multiclass target

df['Value'] = 1
y = df[['TrackID', 'Class', 'Value']].pivot(index = 'TrackID', columns = 'Class', values = 'Value')
y = y.fillna(0)

y.columns = labels.columns

# Prepare data

# Function to convert one hot encoded labels into single binary string

def binary_combinations(y):

    output = []

    for i in range(len(y)):
        vals = y.iloc[i].astype(int).tolist()
        binary = ''.join([str(x) for x in vals])
        integer = int(binary, 2)
        output.append(integer)

    return output

# Remove observations belonging to classes with few representatives

y['Binary_Combination'] = binary_combinations(y)
df_imbalance = y.groupby('Binary_Combination').agg(Count = ('Binary_Combination', 'count')).reset_index()
df_imbalance = df_imbalance[df_imbalance.Count > 10]
y = y[y.Binary_Combination.isin(df_imbalance.Binary_Combination)].drop(columns = 'Binary_Combination')

# Drop duplicates and remove unnecessary variables

X = df.drop_duplicates(subset = 'TrackID')
X = X[X.TrackID.isin(y.index)]

X.index = X.TrackID
X = X.drop(columns = ['Class', 'TrackID', 'Value'])

# count y variable combinations

y_sub = y[y.sum(axis = 1) > 1]

target_combinations = [' '.join(list(y_sub.columns[y_sub.iloc[i,:] == 1])) for i in range(len(y_sub))]

pd.Series(target_combinations).value_counts().to_clipboard()

# Split data into stratified test train sets

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# Create preprocessor step to scale and encode variables

categorical_features = ["key", "mode", "time_signature"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

numeric_features = list(set(X_train.columns).difference(set(categorical_features)))
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

# Define first classifier - Random Forest + Classifer Chain
# Hyper-parameters entered from later tuning step

rf = RandomForestClassifier(class_weight='balanced',
                            max_depth=50,
                            min_samples_leaf=2,
                            min_samples_split=10,
                            n_estimators=2000
                            )
chain = ClassifierChain(rf, order='random', random_state=0)

clf = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("classifier", chain)]
)

# Fit Random Forest Chain model

chain_model = clf.fit(X_train, y_train)

# Evaluate against test set

pred_y = chain_model.predict(X_test)

cmat = multilabel_confusion_matrix(y_test, pred_y)

hamming_loss(y_test, pred_y)

rf_report = classification_report(y_test, pred_y, target_names = labels.columns.to_list(), output_dict=True)
#pd.DataFrame(rf_report).transpose().to_clipboard()

# Random Forest Chain Hyperparameter Tuning
# Commented out due to lengthy run time - the optimal values have been input into the model above

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# #
# random_grid = {'classifier__base_estimator__n_estimators': n_estimators,
#                 'classifier__base_estimator__max_depth': max_depth,
#                 'classifier__base_estimator__min_samples_split': min_samples_split,
#                 'classifier__base_estimator__min_samples_leaf': min_samples_leaf}
#
# rf_random = RandomizedSearchCV(estimator = clf,
#                                 param_distributions = random_grid,
#                                 n_iter = 100,
#                                 cv = 3,
#                                 verbose=2,
#                                 random_state=0,
#                                 n_jobs = -1)
#
# rf_random.fit(X_train, y_train)

# rf_random.best_estimator_

# Second Model - Neural Network

# Define network
# Number of neurons and learning rate from results of later hyperparameter tuning section

def baseline_model():
    model = Sequential()
    model.add(Dense(500, input_dim=28, activation='relu'))
    model.add(Dense(11, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate = 0.01),
                  metrics = HammingLoss(mode='multilabel'))

    return model

# Define early stopping, halting with no improvement after 10 iterations

es = EarlyStopping(monitor='loss', verbose=1, patience=10)

# Create pipeline

estimator = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=32, callbacks=[es])

neural_clf = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("classifier", estimator)]
)

# Run model multiple times and store output

epochs = []
ce_loss = []
h_loss = []
models = dict()

for i in range(10):

    neural_clf.fit(X_train, y_train)
    epochs.append(neural_clf[1].current_epoch)
    ce_loss.append(neural_clf[1].history_['loss'][-1])
    h_loss.append(neural_clf[1].history_['hamming_loss'][-1])
    models[i] = neural_clf

df_nn = pd.DataFrame({'Model':list(range(10)),
                      'Epochs':epochs,
                      'Cross_Entropy_Loss':ce_loss,
                      'Hamming_Loss':h_loss})

#df_nn.to_clipboard()

# Evaluate best performing model against the test set

pred_y = models[df_nn.Hamming_Loss.idxmin()].predict(X_test)

nn_report = classification_report(y_test, pred_y, target_names = labels.columns.to_list(), output_dict=True)

#pd.DataFrame(nn_report).transpose().to_clipboard()

hamming_loss(y_test, pred_y)

# Hyperparameter tuning
# Commented out due to long run time - the optimal values have been input in the previous step
#
# def model_builder(hp):
#      # create model
#      model = Sequential()
#      hp_units = hp.Int('units', min_value=10, max_value=1000, step=10)
#      model.add(Dense(hp_units, input_dim=28, activation='relu'))
#      model.add(Dense(11, activation='sigmoid'))
#
#      # Compile model
#
#      hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
#
#      model.compile(loss='binary_crossentropy',
#                    optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
#                    metrics=HammingLoss(mode='multilabel'))
#
#      return model
#
# X_train_processed = preprocessor.fit_transform(X_train)
#
# tuner = kt.Hyperband(model_builder,
#                      max_epochs=10,
#                      factor=3,
#                      objective=kt.Objective("hamming_loss", direction="min"))
#
# tuner.search(X_train_processed, y_train, epochs=50, validation_split=0.2)
# best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
#
# best_hps.get('units')
# best_hps.get('learning_rate')
