from read_mnist import read_mnist
from rvm_multiclass import rvm


data_dir = '/Users/ronan/Desktop/ml. advanced/project/tipping-rvm-master/MNIST/'

train_images, train_labels = read_mnist(data_dir + 'train-images.idx3-ubyte',
                                        data_dir + 'train-labels.idx1-ubyte',
                                        size_output=12000)
test_images, test_labels = read_mnist(data_dir + 't10k-images.idx3-ubyte',
                                      data_dir + 't10k-labels.idx1-ubyte',
                                      size_output=10000)

rvm(train_x=train_images, train_y=train_labels, test_x=test_images, test_y=test_labels)
