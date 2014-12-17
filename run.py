import numpy as np
import pandas as p
import pickle
import os
import time
import json
from matplotlib import pyplot as plt
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier

try:
    input = raw_input
except NameError:
    pass

memory_threshold = .98
number_per_hour = 100
user_cols = [-2, 3, 5, 6, 7]
system_cols = [-2, 1, 2, 3]


def transform_t(arr, col):
    for entry in arr:
        sp = entry[col].split(":")
        sec = (int(sp[0]) * 3600 + int(sp[1]) * 60 + int(sp[2])) / 3600
        entry[col] = sec


def get_Xy(path, mem_thresh, num_prob):
    user_arr = np.array(p.read_csv(os.path.join(path, "dump_user.csv"),
                                   index_col=None, header=None))
    system_arr = np.array(p.read_csv(os.path.join(path, "dump_system.csv"),
                                     index_col=None, header=None))

    user_arr = user_arr[:, user_cols]
    system_arr = system_arr[:, system_cols]

    transform_t(user_arr, 0)
    transform_t(system_arr, 0)

    mem_perc = system_arr[:, 2] / (system_arr[:, 2] + system_arr[:, 3])
    system_times = system_arr[:, 0].astype(np.int)
    user_times = user_arr[:, 0].astype(np.int)

    trouble_times = np.unique(system_times[mem_perc > mem_thresh])
    indexes = np.arange(user_arr.shape[0])
    y = np.zeros(user_arr.shape[0])

    for t in trouble_times:
        trouble_users = user_arr[user_times == t]
        trouble_indexes = indexes[user_times == t]

        mx = np.argsort(trouble_users[:, 3])[-num_prob:]
        y[trouble_indexes[mx]] = 1

    X = user_arr[:, 1:].astype(np.float64)

    return X, y


def process_single(message):
    arr = np.array(message.split(","))
    arr = arr[user_cols]
    return arr[1:].astype(np.float64)


def process():
    X_train = []
    y_train = np.array([])
    for root, direcs, fi in os.walk("data"):
        for direc in direcs:
            print(direc)
            X_tmp, y_tmp = get_Xy(os.path.join(root, direc),
                                  memory_threshold, number_per_hour)
            X_train.append(X_tmp)
            y_train = np.append(y_train, y_tmp)

    X_train = np.vstack(X_train)

    return X_train, y_train


def fit():
    X_train, y_train = process()

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train.astype(np.float64))

    clf = SGDClassifier(n_iter=10, average=True, loss="log")
    clf.fit(X_train, y_train)

    pickle.dump(clf, open("classifier.p", "wb"))
    pickle.dump(ss, open("scaler.p", "wb"))

# use to predict full file


def full_predict():
    clf = pickle.load(open("classifier.p", "rb"))
    ss = pickle.load(open("scaler.p", "rb"))

    # use this code to predict full set
    X, y = get_Xy("data/2014-07-21/",
                  memory_threshold, number_per_hour)

    X = ss.fit_transform(X)

    print("score: %.8f" % clf.score(X, y))


def graph_performance():
    num_values = 20
    X_train, y_train = process()
    increment = X_train.shape[0] / num_values
    clf = SGDClassifier(n_iter=10, average=True, loss="log")

    times = []
    num_samples = []
    for i in range(1, num_values):
        print("training %d" % i)
        num_samples.append(i * increment)
        X_slice = X_train[:i * increment]
        y_slice = y_train[:i * increment]
        st = time.time()
        clf.fit(X_slice, y_slice)
        end = time.time()
        times.append(end - st)

    plt.close("all")
    plt.plot(num_samples, times, label='Online ASGD')
    plt.ylabel("training time (seconds)")
    plt.xlabel("number of log entries")
    plt.title('Learning computation time')
    plt.legend(loc='upper left')
    plt.show()


def loop():
    clf = pickle.load(open("classifier.p", "rb"))
    ss = pickle.load(open("scaler.p", "rb"))

    while True:
        log_message = input('Enter a user log message (type exit to exit):\n')
        if log_message == "exit":
            print("Thanks for trying this demo out!")
            break
        else:
            print("")
            try:
                entry = process_single(log_message)
                entry = ss.transform(entry)
                pred = clf.predict_proba(entry)
                print("The probability that this is a dangerous user is %0.8f"
                      % pred[0, 1])
            except:
                print("doesn't look like the message "
                      "was in the correct format")
            print("")

def to_json():
    clf = pickle.load(open("classifier.p", "rb"))
    ss = pickle.load(open("scaler.p", "rb"))

    obj = {}

    obj["weights"] = clf.coef_.tolist()[0]
    obj["scale"] = {}
    obj["scale"]["mean"] = ss.mean_.tolist()
    obj["scale"]["std"] = ss.std_.tolist()

    json.dump(obj, open("web/data/model.json", "w"))

if __name__ == "__main__":
    # comment this out after computing so you don't have to keep processing and
    # fitting
    # fit()

    # uncoment to score a full log file
    # full_predict()

    # try the demo loop
    # loop()

    # graph time performance
    # graph_performance()

    # write to json
    to_json()
