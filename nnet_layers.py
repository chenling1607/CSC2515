# Code by Navdeep Jaitly, 2013
# Email: ndjaitly@gmail.com

from numpy import sqrt, isnan, Inf, dot, zeros, exp, log, sum, transpose
from numpy.random import randn

SIGMOID_LAYER = 0
SOFTMAX_LAYER = 1

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

class layer_definition(object):
    def __init__(self, name, layer_type, input_dim, num_units, wt_sigma):
        self.name, self.layer_type, self.input_dim, self.num_units, \
          self.wt_sigma  =  name, layer_type, input_dim, num_units, wt_sigma


class layer(object):
    def __init__(self, name):
        self.name = name

    @property
    def shape(self):
        return self._wts.shape

    @property
    def num_hid(self):
        return self._wts.shape[1]

    @property
    def num_dims(self):
        return self._wts.shape[0]

    def create_params(self, layer_def):
        input_dim, output_dim, wt_sigma = layer_def.input_dim, \
                        layer_def.num_units, layer_def.wt_sigma

        self._wts = randn(input_dim, output_dim) * wt_sigma
        self._b = zeros((output_dim, 1))

        self._wts_grad = zeros(self._wts.shape)
        self._wts_inc = zeros(self._wts.shape)

        self._b_grad = zeros(self._b.shape)
        self._b_inc = zeros(self._b.shape)

        self.__num_params = input_dim*output_dim

    def add_params_to_dict(self, params_dict):
        params_dict[self.name + "_wts"] = self._wts.copy()
        params_dict[self.name + "_b"] = self._b.copy()

    def copy_params_from_dict(self, params_dict):
        self._wts = params_dict[self.name + "_wts"].copy()
        self._b = params_dict[self.name + "_b"].copy()
        self.__num_params = self._wts.shape[0] * self._wts.shape[1]
        self._wts_inc = zeros(self._wts.shape)
        self._b_inc = zeros(self._b.shape)

    def apply_gradients(self, momentum, eps, l2=.0001):
        """ NEED TO IMPLEMENT 
        """
        raise Exception, "Unimplemented functionality"

    def back_prop(self, act_grad, data):
        ''' 
        NEED TO IMPLEMENT. 
        Feel free to add member variables.
        back prop activation grad, and compute gradients. 
        '''
        raise Exception, "Unimplemented functionality"
        return input_grad
 

class sigmoid_layer(layer):
    pass 

    def fwd_prop(self, data):
        """ NEED TO IMPLEMENT 
        """
        #print ('Data', data.shape[0], 'X', data.shape[1])

        bias = zeros((self._b.shape[0], data.shape[1]))
        for i in range(data.shape[1]):
            bias[:,i] = self._b[:,0]

        activation = dot(transpose(self._wts), data) + bias

        probs = sigmoid(activation)
        return probs

    def compute_act_grad_from_output_grad(self, output, output_grad):
        """ NEED TO IMPLEMENT 
        """
        
        return act_grad

 
class softmax_layer(layer):
    pass

    def fwd_prop(self, data):
        """ NEED TO IMPLEMENT 
        """
        bias = zeros((self._b.shape[0], data.shape[1]))
        for i in range(data.shape[1]):
            bias[:,i] = self._b[:,0]

        activation = dot(transpose(self._wts), data) + bias

        probs = sigmoid(activation)
        for j in range(probs.shape[1]):
            s = 0
            #for i in range(probs.shape[0]):
            #    s += probs[i][j]
            s = sum(probs[:,j])
            for i in range(probs.shape[0]):
                probs[i][j] /= s

        return probs

    def compute_act_gradients_from_targets(self, targets, output):
        """ NEED TO IMPLEMENT 
        """
        raise Exception, "Unimplemented functionality"
        return act_grad


    @staticmethod 
    def compute_accuraccy(probs, label_mat):
        num_correct = sum(probs.argmax(axis=0) == label_mat.argmax(axis=0))
        log_probs = sum(log(probs) * label_mat)
        print (probs.shape, label_mat.shape)
        return num_correct, log_probs

def create_empty_nnet_layer(name, layer_type):
    if layer_type == SIGMOID_LAYER:
        layer = sigmoid_layer(name)
    elif layer_type == SOFTMAX_LAYER:
        layer = softmax_layer(name)
    else:
        raise Exception, "Unknown layer type"
    return layer

def create_nnet_layer(layer_def):
    layer = create_empty_nnet_layer(layer_def.name, layer_def.layer_type)
    layer.create_params(layer_def)
    return layer
