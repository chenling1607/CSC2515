import numpy as np, scipy.io, sys, argparse
import scipy.spatial.distance as ssd
from PCA import *

def knn(k, dtrain, dtest, dtr_label, dist=1):
    ''' k-nearest neighbors '''
 
    # initialize list to store predicted class
    pred_class = []
    # for each instance in data testing,
    # calculate distance in respect to data training
    for ii, di in enumerate(dtest.T):
        distances = []  # initialize list to store distance
        for ij, dj in enumerate(dtrain.T):
            # calculate distances
            distances.append((calc_dist(di,dj,dist), ij))
        # k-neighbors
        k_nn = sorted(distances)[:k]
        #k_nn = sorted(distances)
        
        # predict the class for the instance
        #print 'k_nn', k_nn
        #print 'dtr', dtr_label
        pred_class.append(classify(k_nn, dtr_label))
        #pred_class.append(non_para_classify(k_nn, dtr_label))

        sys.stdout.write('\r')
        sys.stdout.write("Percentage of Completion: %s   " % (ii/22))
        sys.stdout.flush()
    # return prediction class
    return pred_class
 
def calc_dist(di,dj,i=1):
    ''' Distance calculation for every
        distance functions in use'''
    if i == 2:
        return ssd.euclidean(di,dj) # built-in Euclidean fn
    elif i == 1:
        return ssd.cityblock(di,dj) # built-in Manhattan fn
    elif i == 3:
        return ssd.cosine(di,dj)    # built-in Cosine fn
 
def classify(k_nn, dtr_label):
    ''' Classify instance data test into class'''
 
    dlabel = []
    for dist, idx in k_nn:
        # retrieve label class and store into dlabel
        dlabel.append(int(dtr_label[idx]))
 
    # return prediction class
    dlabel = np.reshape(dlabel, len(dlabel))
    #dlabel.astype(np.int64)
    return np.argmax(np.bincount(dlabel))

def non_para_classify(k_nn, dtr_label):
    ''' Classify instance data test into class'''
 
    contribute = np.zeros(44)
    for dist, idx in k_nn:
        # retrieve label class and store into dlabel
        contribute[int(dtr_label[idx])] += 1.0 / (dist**4 + 1e-32)
 
    
    #dlabel.astype(np.int64)
    #print contribute
    #print np.argmax(contribute)
    return np.argmax(contribute)

def evaluate(result):
    ''' Evaluate the prediction class'''
 
    # create eval result array to store evaluation result

    eval_result = np.zeros(2,int)
    for x in result:
        # increment the correct prediction by 1
        if x == 0:
            eval_result[0] += 1
        # increment the wrong prediction by 1
        else:
            eval_result[1] += 1
    # return evaluation result
    return eval_result
 

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--db_path', default = '/Users/chenling/Desktop/A2/knn_subset.mat', help='Path to database')

	arguments = parser.parse_args()

	params_dict = scipy.io.loadmat(arguments.db_path)


	K = [1]

	train_targets = params_dict['train_targets']  # 4400 1
	test_targets = params_dict['test_targets']   # 2200 1
	train_data = params_dict['train_data']   # 23 4400
	test_data = params_dict['test_data']   # 23 2200

	dist_fn = [1]

	#train_data, test_data = PCA(20, train_data, test_data)

	indices_to_corrupt = np.random.permutation(4400)[:440]
	train_targets[indices_to_corrupt, 0] = np.random.multinomial( \
		1, [1/44.] * 44, size = indices_to_corrupt.size).argmax(axis = 1)

    # run knn classifier for each k and distance function
	for i in range(len(K)):
    # classification result for each distance function
		results = []
		for j in range(len(dist_fn)):
            # predict the data test into class
			pred_class = knn(K[i], train_data, test_data, train_targets, dist_fn[j])

			sys.stdout.write('\r')
			sys.stdout.flush()
			test_targets = np.reshape(test_targets, test_targets.shape[0])
			eval_result = evaluate(pred_class-test_targets)
			print  '                                           '
			print  '                                           '
 			print 'K is', K[i], 'Metric is ',  dist_fn[j]
			#print the classification result into the screen
			print 'Test Error: ', float(eval_result[1]) / float(eval_result[0] + eval_result[1])
	
