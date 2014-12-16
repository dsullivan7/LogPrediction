import numpy as np
import pandas as p
import pickle
import os
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier

memory_threshold = .98
number_per_hour = 100

def train():
	X_train, y_train = get_Xy("data/2014-07-21/", memory_threshold, number_per_hour)
	X_test, y_test = get_Xy("data/2014-09-30/", memory_threshold, number_per_hour)
	# X_test, y_test = get_Xy("data/2014-12-14/", memory_threshold, number_per_hour)

	clf = SGDClassifier(n_iter=1000, average=True)
	# clf = Birch()
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)

	print("score: %1.6f" % np.mean(pred == y_test))


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

	user_arr = user_arr[:, [-2, 3, 5, 6, 7]]
	system_arr = system_arr[:, [-2, 1, 2, 3]]

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

	X = user_arr[:, 1:]
	ss = StandardScaler()
	X = ss.fit_transform(X.astype(np.float64))

	return X, y

def loop(reload=False):
	X_train = []
	y_train = np.array([])
	if reload:
		for root, direcs, fi in os.walk("data"):
			for direc in direcs:
				print(direc)
				X_tmp, y_tmp = get_Xy(os.path.join(root, direc),
									  memory_threshold, number_per_hour)
				X_train.append(X_tmp)
				y_train = np.append(y_train, y_tmp)

		X_train = np.vstack(X_train)

		clf = SGDClassifier(n_iter=1000, average=True)
		clf.fit(X_train, y_train)

		pickle.dump(clf, open("classifier.p", "wb"))

	clf = pickle.load(open("classifier.p", "rb"))


	X_tmp, y_tmp = get_Xy("data/2014-10-26/", memory_threshold, number_per_hour)
	print("score %1.8f" % clf.score(X_tmp, y_tmp))


if __name__ == "__main__":
	loop(False)
