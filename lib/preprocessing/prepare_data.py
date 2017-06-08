# /usr/local/bin/python3.5
# Made by Bjoernar Remmen
# Short is better
import h5py
import numpy as np

def addaxis(image):
    return image[np.newaxis, np.newaxis, :, :]

def get_ids(label_array):
    id_label = {}
    counter = 0
    for labels in label_array:
        for label in labels:
            if label not in id_label:
                id_label[label] = counter
                counter += 1
    return id_label, counter+1

def save_h5_file(name,directory,images, label_array):
    filename_h5 = directory + name + ".h5"
    filename_txt = directory + name + ".txt"

    #label_index, array_len = get_ids(label_array)

    new_labels = []
    #for i in range(len(images)):
     #   new_labels.append([label_index[word] for word in label_array[i]])
       #images[i] = images[i,np.newaxis,:,:]

    with h5py.File(filename_h5, "w") as f:
        f['data'] = images
        f['label'] = label_array

    with open(filename_txt, "w") as text_file:
        text_file.write(filename_h5)

    return filename_txt,filename_h5










