import os

import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.nonlinearities import tanh
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle

def wheel_base(bl, br, w):
    r=0.2
    

class BCNet:

    def __init__(self):
        self.params = {
            'inp_size': 10,
            'outp_size': 5,
            'hidden': 100
        }



    def normalize_data(self, inp):
        self.params['inp_mean'] = inp.mean(axis=0)
        self.params['inp_std'] = inp.std(axis=0)
        # import pdb; pdb.set_trace()
        return preprocessing.scale(inp)
        

    def create_net(self):
        if 'net1' not in vars(self):
            self.net1=NeuralNet(
                layers=[  # three layers: one hidden layer
                    ('input', layers.InputLayer),
                    ('hidden', layers.DenseLayer),
                    ('output', layers.DenseLayer),
                    ],
                # layer parameters:
                input_shape=(None, self.params['inp_size']),  # 96x96 input pixels per batch
                # input_shape=(None, 96*96),  # 96x96 input pixels per batch
                hidden_num_units=self.params['hidden'],  # number of units in hidden layer
                output_nonlinearity=None,  # output layer uses identity function
                hidden_nonlinearity=tanh,
                # output_num_units=1,  # 30 target values
                output_num_units=self.params['outp_size'],  # 30 target values
                # output_num_units=30,  # 30 target values

                # optimization method:
                update=nesterov_momentum,
                update_learning_rate=0.01,
                update_momentum=0.9,
                objective_l2=0.001,

                regression=True,  # flag to indicate we're dealing with regression problem
                max_epochs=400,  # we want to train this many epochs
                verbose=1,
                )
    
    def get_train_data(self, fname):
        np.random.seed(113)
        with open(fname,'r') as f:
            data = pickle.load(f)

        inp = data['observations']
        outp = data['actions']
        self.params['inp_size'] = data['inp_size']
        self.params['outp_size'] = data['outp_size']

        inp, outp = shuffle(inp, outp, random_state=42)  # shuffle train data
        return (inp, np.squeeze(outp))


    def load_model(self, fname):
        params_name = fname+'__params'
        with open(params_name,'r') as f:
            self.params = pickle.load(f)
        self.create_net()
        self.net1.load_params_from(fname)

    def train_new_model(self, data_fname='test1.csv'):
        X, y = self.get_train_data(data_fname)
        X = self.normalize_data(X)

        print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
            X.shape, X.min(), X.max()))
        print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
            y.shape, y.min(), y.max()))
        self.create_net()
        self.net1.fit(X,y)

    def save_params_to(self, fname):        
        print 'Saving model parameters to file %s' % fname
        self.net1.save_params_to(fname)
        params_name = fname+'__params'
        with open(params_name,'w') as f:
            pickle.dump(self.params,f)

# This takes the data from sname (a time series with the format in data.csv), predicts vx, vy, vz based on previous times and current bl,br,fl,fr ,
# and writes it to dname 

    def predict_for(self, inp):
        m = self.params['inp_mean']
        s = self.params['inp_std']
        s[s==0] = 1
        X = (inp - m) / s
        return self.net1.predict(X[None,:])
        

    def plot_velocity(self, fname, vel_index=(7,8,9), fw_index=(15,16) ):
        a = np.loadtxt(open(fname,"rb"),delimiter=",")
        vels = a[:,vel_index]
        z = np.linalg.norm(vels, axis=1)
        x = a[:,fw_index[0]]
        y = a[:,fw_index[1]]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(x)):
            ax.scatter(x[i],y[i],z[i], c='r', marker='o')

        ax.set_xlabel('Front Left')
        ax.set_ylabel('Front Right')
        ax.set_zlabel('Norm of velocity')
        plt.show()




# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('training_file', type=str)
#     parser.add_argument('network_file', type=str)
#     parser.add_argument('--hidden', type=int, default=100,
#         help='Size of hidden layer')
#     parser.add_argument('--dagger', type=int, default=0,
#         help='Number of dagger iterations')
#     # parser.add_argument('--render', action='store_true')
#     # parser.add_argument("--max_timesteps", type=int)
#     args = parser.parse_args()

#     a = BCNet()
#     a.params['hidden']=args.hidden
#     a.train_new_model(args.training_file)

#     for i in range(args.dagger):

#     a.save_params_to(args.network_file)


# if __name__ == '__main__':
#     main()
