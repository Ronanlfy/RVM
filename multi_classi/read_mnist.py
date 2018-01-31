import numpy as np


"""
   Function to parse dataset from http://yann.lecun.com/exdb/mnist/
   arguments :
       image_file : path to the images file
       labels_file : path to the labels
       size_output : number of images to parse
   outputs:
       (ims,labs):
           ims : a nd-array of size (size_output,height*width) coding the images
           labs : a nd-array of size (size_output) coding the labels
"""


def read_mnist(image_file, labels_file, size_output=10000):
    with open(image_file,'rb') as file:
        with open(labels_file,'rb') as labs:
            magic_number = file.read(4)
            im_number = int.from_bytes(file.read(4),byteorder='big')
            im_width = int.from_bytes(file.read(4),byteorder='big')
            im_height = int.from_bytes(file.read(4),byteorder='big')
            labs.read(8)
            output_ims = np.zeros((size_output,im_width*im_height))
            output_labels = np.zeros(size_output, dtype=int)
            for k in range(min(size_output,im_number)):
                for i in range(im_width):
                    for j in range(im_height):
                        output_ims[k,i*j+j] = int.from_bytes(file.read(1),byteorder='big')
                output_labels[k] = int.from_bytes(labs.read(1),byteorder='big')
    return output_ims, output_labels
