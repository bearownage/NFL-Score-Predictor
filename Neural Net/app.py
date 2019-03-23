import pickle
import numpy as np         # linear algebra
import sklearn as sk       # machine learning
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing  # for standardizing the data
import pandas as pd
import seaborn as sns      # visualization tool
import matplotlib.pyplot as plt   # for plotting
import seaborn as sns      # visualization tool
import tensorflow as tf    # for creating neural networks
from tensorflow import keras   # an easier interface to work with than tensorflow
import scrape
import os, sys

def openAllTeamsFromSeason(season):
    with open("teams" + str(season) + ".pickle", 'rb') as handle:
        data = pickle.load(handle)
        return data

def openTeamFromSeason(team,season):
    with open("teams" + str(season) + ".pickle", 'rb') as handle:
        data = pickle.load(handle)
        return data[team]

def openSeason(season):
    with open("season" + str(season) + ".pickle", 'rb') as handle:
        data = pickle.load(handle)
        return data

def getStat(team,stat):
    return float(team[stat])

def calculate(home,away):
    #['g', 'points', 'total_yards', 'plays_offense', 'yds_per_play_offense',
    #'turnovers', 'fumbles_lost', 'first_down', 'pass_cmp', 'pass_att', 'pass_yds',
    #'pass_td', 'pass_int', 'pass_net_yds_per_att', 'pass_fd', 'rush_att', 'rush_yds',
    # 'rush_td', 'rush_yds_per_att', 'rush_fd', 'penalties', 'penalties_yds', 'pen_fd',
    #'score_pct', 'turnover_pct', 'exp_pts_tot'])
    stats = ["turnover_pct","exp_pts_tot","yds_per_play_offense"]
    data = []
    for stat in stats:
      data.append(getStat(home,stat))
    data.append(1)
    for stat in stats:
      data.append(getStat(away,stat))
    data.append(0)
    return data

def buildVectors():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(2000,2019):
        teams_for_year = openAllTeamsFromSeason(i)
        head_to_head = openSeason(i)
        for game in head_to_head:
            winner = teams_for_year[game[0]]
            loser = teams_for_year[game[1]]
            if i <= 2017:
                if game[2] == 1:
                    # AWAY WIN
                    x_train.append(calculate(loser,winner))
                    y_train.append(1)
                else:
                    x_train.append(calculate(winner,loser))
                    y_train.append(0)
            else:
                if game[2] == 1:
                    # AWAY WIN
                    x_test.append(calculate(loser,winner))
                    y_test.append(1)
                else:
                    x_test.append(calculate(winner,loser))
                    y_test.append(0)
    return x_train, y_train, x_test, y_test


def network(showPlot=False):
    Dense = keras.layers.Dense
    Activation = keras.layers.Activation
    to_categorical = keras.utils.to_categorical
    Sequential = keras.Sequential

    x_train, y_train, x_test, y_test = buildVectors()
    # Standardize the data.
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)

    # We also need to transform the test set, using the same means and standard deviations
    # that were calculated from and used to transform the training set data.
    x_test = scaler.transform(x_test)
    train_hot_labels = to_categorical(y_train, num_classes = 600)
    test_hot_labels = to_categorical(y_test, num_classes = 600)
    # Instantiate a new neural network model
    model = Sequential()

    # Add some layers.
    # Fist the input layer, which has 7 values, is connected to hidden layer 5, with 100 nodes (neurons).
    model.add(Dense(1400, activation='sigmoid', input_dim = 8))
    # Layer 2, hidden layer
    model.add(Dense(900, activation='sigmoid'))
    # Layer 3, output layer
    model.add(Dense(600, activation = 'softmax', ))

    # Compile the NN model, defining the optimizer to use, the loss function, and the metrics to use.
    # These settings are appropriate for a multiple-class classification task.
    model.compile(optimizer = 'rmsprop',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    # Train the model, iterating on the data in batches of 64 sample, over 8 epochs
    history = model.fit(x_train, train_hot_labels,
                        validation_split = 0.25,
                        epochs = 8,
                        batch_size = 64)

    # Evaluate the model's performance
    train_loss, train_acc = model.evaluate(x_train, train_hot_labels)
    test_loss, test_acc = model.evaluate(x_test, test_hot_labels)

    print('Training set accuracy:', train_acc)
    print('Test set accuracy:', test_acc)

    if showPlot:
        print("Showing Plots")
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

if __name__== "__main__":
    # UNCOMMENT THIS LINE IF IT THROWS ERROR 15 REGARDING libiomp5.dylib
    # os.environ['KMP_DUPLICATE_LIB_OK']='True'
    if "scrape" in sys.argv:
        scrape.scrapeTeamOffenses()
        scrape.scrapeWeeklyResults()
    network(True if "plot" in sys.argv else False)
