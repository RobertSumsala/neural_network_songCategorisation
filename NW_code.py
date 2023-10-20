# ZADANIE 01 - Robert Sumsala



print()
print("- - - - - - - ZADANIE 01 - - - - - - -")
print()
print()

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from keras.src.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.src.callbacks import EarlyStopping
from keras.src.utils import to_categorical
from pyasn1.compat.octets import null


# Allow printing more columns
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# Do not show pandas warnings
pd.set_option('mode.chained_assignment', None)

# Load the dataset with Song specs
df = pd.read_csv('./data/zadanie1_dataset.csv')

# -----------------------------------------------------------------------------------------------------------------------------------------------

# Print min and max values of columns before removing outliers
print("*"*30, "Before removing outliers", "*"*30)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))

# Removing outliers from: duration_ms (based on min), danceability (based on max), loudness (based on max),
df = df[df['duration_ms'] >= 0]
df = df[df['danceability'] <= 1]
df = df[df['loudness'] <= 0]

# Print min and max values of columns - checking if the values are from correct intervals
print("*"*30, "After removing outliers", "*"*30)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))


# Count missing values in columns
print("*"*30, "Missing values", "*"*30)
print(f"Length of the dataset: {len(df)}")
print(df.isnull().sum())

# Remove columns with missing values (and are not needed)
df.drop(columns=['popularity', 'number_of_artists', 'top_genre'], inplace=True)
# Dropping columns that are not needed
df.drop(columns=['url'], inplace=True)

# Checking the missing values count after removing the columns
print("*"*30, "Missing values after removing them", "*"*30)
print(f"Lenght of dataset: {len(df)}")
print(df.isnull().sum())


# save unlabeled df to df2 for the third part of assignment
df2 = df.copy()


# Print column types
print("*"*30, "Column types", "*"*30)
print(df.dtypes)

# Label encoding for names, genres, filtered genres and emotions
le = LabelEncoder()
df['name_encoded'] = le.fit_transform(df['name'])
df['genres_encoded'] = le.fit_transform(df['genres'])
df['filtered_genres_encoded'] = le.fit_transform(df['filtered_genres'])
df['emotion_encoded'] = le.fit_transform(df['emotion'])

print("*"*30, "Label encoding", "*"*30)
print(df[['name', 'name_encoded']].head(10))
print()
print(df[['genres', 'genres_encoded']].head(10))
print()
print(df[['filtered_genres', 'filtered_genres_encoded']].head(10))
print()
print(df[['emotion', 'emotion_encoded']].head(10))

# removing columns that now has their encoded values in a different column
df.drop(columns=['name', 'genres', 'filtered_genres', 'emotion'], inplace=True)


# Splitting dataset into X and y
X = df.drop(columns=['emotion_encoded'])
y = df['emotion_encoded']

# Splitting dataset into train, valid and test
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.25, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5, random_state=42)


# Printing stats before scaling
# Print dataset shapes
print("*"*30, "Dataset shapes", "*"*30)
print(f"X_train: {X_train.shape}")
print(f"X_valid: {X_valid.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_valid: {y_valid.shape}")
print(f"y_test: {y_test.shape}")

# Plot histograms before scaling
X_train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms before scaling/standardizing')
plt.show()

# Print min and max values of columns
print("*"*30, "Before scaling/standardizing", "*"*30)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))


# we're going to use min max scaling on the values that aren't in the range <0,1> since most of the attribute are

# Scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid_test)
X_test = scaler.transform(X_test)

# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)


# Print data after scaling
# Plot histograms after scaling
X_train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms after scaling')
plt.show()

# Print min and max values of columns
print("*"*30, "After scaling", "*"*30)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))


# Train MLP model to predict emotion
print("*"*30, "MLP", "*"*30)
clf = MLPClassifier(
    hidden_layer_sizes=(50, 25, 10),
    random_state=1,
    max_iter=250,
    validation_fraction=0.2,
    early_stopping=False,
    learning_rate='adaptive',
    learning_rate_init=0.001,
).fit(X_train, y_train)


# Predict on train set
y_pred = clf.predict(X_train)
print('MLP accuracy on train set: ', accuracy_score(y_train, y_pred))
cm_train = confusion_matrix(y_train, y_pred)

# Predict on test set
y_pred = clf.predict(X_test)
print('MLP accuracy on test set: ', accuracy_score(y_test, y_pred))
cm_test = confusion_matrix(y_test, y_pred)


# Confusion Matrix for Sklearn simple train
class_names = list(le.inverse_transform(clf.classes_))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on train set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on test set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()


# -----------------------------------------------------------------------------------------------------------------------------------------------


# EDA
# HEATMAPs - using pandas
plt.figure(figsize=(10, 8))
plt.hist2d(df['energy'], df['tempo'], bins=(300, 300), cmap='viridis')
plt.colorbar(label='Frequency')
plt.xlabel('Energy')
plt.ylabel('Tempo')
plt.title('Relation between ENERGY and TEMPO')
plt.show()

plt.figure(figsize=(10, 8))
plt.hist2d(df['energy'], df['loudness'], bins=(300, 300), cmap='viridis')
plt.colorbar(label='Frequency')
plt.xlabel('Energy')
plt.ylabel('Loudness')
plt.title('Relation between ENERGY and LOUDNESS')
plt.show()

plt.figure(figsize=(10, 8))
plt.hist2d(df['instrumentalness'], df['liveness'], bins=(300, 300), cmap='viridis')
plt.colorbar(label='Frequency')
plt.xlabel('Instrumentalness')
plt.ylabel('Liveness')
plt.title('Relation between INSTRUMENTALNESS and LIVENESS')
plt.show()

plt.figure(figsize=(10, 8))
plt.hist2d(df['tempo'], df['danceability'], bins=(300, 300), cmap='viridis')
plt.colorbar(label='Frequency')
plt.xlabel('Tempo')
plt.ylabel('Danceability')
plt.title('Relation between TEMPO and DANCEABILITY')
plt.show()

plt.figure(figsize=(10, 8))
plt.hist2d(df['danceability'], df['emotion_encoded'], bins=(300, 300), cmap='viridis')
plt.colorbar(label='Frequency')
plt.xlabel('Danceability')
plt.ylabel('Emotion')
plt.yticks(range(len(le.classes_)), le.classes_)
plt.title('Relation between DANCEABILITY and EMOTION')
plt.show()


# -----------------------------------------------------------------------------------------------------------------------------------------------


# we'll be using dataframe df2, which is the copy of df before any encoding was done to it (outliers and columns had been removed)

# save original unique string values before encoding for cm
emotion_strings_unique = df2['emotion'].unique().tolist()

# dropping additional columns that are not needed
df2.drop(columns=['genres', 'filtered_genres', 'name'], inplace=True)


# Splitting dataset into X and y
X = df2.drop(columns=['emotion'])
y = df2['emotion']

# # One-hot encoding
y_enc = pd.get_dummies(y)
print(y_enc[0:10])

# Splitting dataset into train, valid and test
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y_enc, shuffle=True, test_size=0.25, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5, random_state=42)


# Before Scaling
# Print min and max values of columns
print("*"*30, "Before scaling", "*"*30)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))

# Apply Min-Max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# After Scaling
# Print min and max values of columns
print("*"*30, "After scaling", "*"*30)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))

# Print dataset shapes after scaling
print("*"*30, "Dataset shapes", "*"*30)
print(f"X_train: {X_train.shape}")
print(f"X_valid: {X_valid.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_valid: {y_valid.shape}")
print(f"y_test: {y_test.shape}")


# Train MLP model in Keras (overtrain with early stopping)
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Early stopping definition
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size=64, callbacks=[early_stopping])


# Evaluation
test_scores = model.evaluate(X_test, y_test, verbose=0)

print("*"*30, "Accuracy", "*"*30)
print(f"Test accuracy: {test_scores[1]:.4f}")
print(f"Train accuracy: {history.history['accuracy'][-1]:.4f}")

# Test cm
# Plot confusion matrix
y_pred_enc = model.predict(X_test)

# Convert the predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred_enc, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Confusion matrix
class_names = emotion_strings_unique

cm = confusion_matrix(y_test_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on test set")
disp.ax_.set(xlabel='PREDICTED', ylabel='TRUE')
plt.show()

# Train cm
# Plot confusion matrix
y_pred_enc = model.predict(X_train)

# Convert the predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred_enc, axis=1)
y_train_classes = np.argmax(y_train, axis=1)

# Confusion matrix
class_names = emotion_strings_unique

cm = confusion_matrix(y_train_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on train set")
disp.ax_.set(xlabel='PREDICTED', ylabel='TRUE')
plt.show()


# Plot loss and accuracy
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Criterion Function')
plt.title('Loss')
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()


# -----------------------------------------------------------------------------------------------------------------------------------------------


# RESOURCES
    # guide for the whole process: codes from seminars (seminar2.py, sem3.py)
    # help with early stopping: https://keras.io/api/callbacks/early_stopping/
    # help with one-hot encoding: https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/
    # additional help: keras documantation and stackoverflow


























