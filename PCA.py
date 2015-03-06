import numpy as np, scipy.io, sys


def PCA(n, train_data, test_data):
	U, s, V = np.linalg.svd(train_data.T)
	Z = np.dot(U[:, :n], np.eye(n)*s[:n])
	Z = Z.T

	print s
	Cov = np.dot(train_data, train_data.T)
	#print 'Cov', Cov.shape
	#print np.dot(Cov, train_data).shape
	Cov = np.linalg.inv(Cov)
	W = np.dot(Cov, train_data).dot(Z.T)
	#print 'W', W.shape
	Y = np.dot(test_data.T, W)
	return Z, Y.T


if __name__ == '__main__':

	params_dict = scipy.io.loadmat('/Users/chenling/Desktop/A2/knn_subset.mat')

	train_targets = params_dict['train_targets']  # 4400 1
	test_targets = params_dict['test_targets']   # 2200 1
	train_data = params_dict['train_data']   # 23 4400
	test_data = params_dict['test_data']   # 23 2200


	train_data, test_data = PCA(5, train_data, test_data)

	print train_data.shape, test_data.shape