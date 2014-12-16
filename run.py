import numpy as np
import pandas as p
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier


def user():
	df = p.read_csv("data/2014-07-21/dump_user.csv", index_col=None, header=None)
	arr = np.array(df)
	arr = arr[:, [3, 5, 6, 7]]
	# arr = arr[:, [5, 9]]
	plotting_array = arr.copy()
	ss = StandardScaler()
	arr = ss.fit_transform(arr.astype(np.float64))
	# import pdb; pdb.set_trace()
	clf = Birch(n_clusters=2)
	print("fitting")
	clf.fit(arr)
	print("done!!")
	colors = ['r', 'g', 'b', 'c', 'm', 'y' 'b']
	plt.close('all')
	for i, label in enumerate(np.unique(clf.labels_)):
		print("plotting %s" % label)
		full = plotting_array[clf.labels_ == label]
		x = full[:, 0]
		y = full[:, 1]
		plt.scatter(x, y, c=colors[i])

	plt.xlabel("memory %")
	plt.ylabel("memory rss (% memory in ram)")
	plt.show()

def system():
	df = p.read_csv("data/2014-12-14/dump_system.csv", index_col=None, header=None)
	arr = np.array(df)
	arr = arr[:, [-2, 2, 3]]

	for entry in arr:
		sp = entry[0].split(":")
		sec = (int(sp[0]) * 3600 + int(sp[1]) * 60 + int(sp[2])) / 3600
		entry[0] = sec

	# arr = arr[:, [5, 6]]
	plotting_array = arr.copy()
	ss = StandardScaler()
	arr = ss.fit_transform(arr.astype(np.float64))
	# import pdb; pdb.set_trace()
	# clf = Birch(n_clusters=2)
	# print("fitting")
	# clf.fit(arr)
	# print("done!!")

	colors = ['r', 'g', 'b', 'c', 'm', 'y' 'b']
	plt.close('all')
	# for i, label in enumerate(np.unique(clf.labels_)):
	# 	print("plotting %s" % label)
	# 	full = plotting_array[clf.labels_ == label]
	# 	x = full[:, 0]
	# 	y = full[:, 1]
	# 	plt.scatter(x, y, c=colors[i])
	full = plotting_array
	x = full[:, 0]
	y = full[:, 1] / (full[:, 1] + full[:, 2])
	plt.scatter(x, y, c='r')

	plt.xlabel("time (hours)")
	plt.ylabel("cpu")
	plt.show()

def user_time():

	ss = StandardScaler()
	arr = ss.fit_transform(arr.astype(np.float64))
	# import pdb; pdb.set_trace()
	# clf = Birch(n_clusters=2)
	print("fitting")
	# clf.fit(arr)
	print("done!!")
	colors = ['r', 'g', 'b', 'c', 'm', 'y' 'b']
	plt.close('all')

	for i, user in enumerate(uusers):
		full = plotting_array[users == user]
		# x = full[:, 0]
		x = full[:, 0]
		y = full[:, 1]
		plt.scatter(x, y, c=colors[0])
		plt.xlabel("memory %")
		plt.ylabel("memory rss")
		plt.show()
		plt.close("all")
		if i > 5:
			break

def train():
	X_train, y_train = get_Xy("data/2014-07-21/", .8, 100)
	X_test, y_test = get_Xy("data/2014-09-30/", .8, 100)

	import pdb; pdb.set_trace()

	clf = SGDClassifier(n_iter=1000, average=True)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	print(pred[pred == 1])
	print("score: ", clf.score(X_test, y_test))


def transform_t(arr, col):
	for entry in arr:
		sp = entry[col].split(":")
		sec = (int(sp[0]) * 3600 + int(sp[1]) * 60 + int(sp[2])) / 3600
		entry[col] = sec

def get_Xy(path, mem_thresh, num_prob):
	user_arr = np.array(p.read_csv(path + "dump_user.csv", index_col=None, header=None))
	system_arr = np.array(p.read_csv(path + "dump_system.csv", index_col=None, header=None))

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

if __name__ == "__main__":
	train()
