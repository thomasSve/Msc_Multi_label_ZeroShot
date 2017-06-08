#!/usr/bin/env python
# Made by Bjoernar Remmen
# Short is better

bad = ["train/6572/0fb2ed69f4bae4e2.jpg",
       "train/6574/a3fea9921acdfbc3.jpg",
       "train/7551/173d25f00beb04ce.jpg",
       "train/7545/c5d7cbb12b608740.jpg",
       "train/7532/b8425ddfe824f141.jpg",
       "train/7551/050c5d8e42bf84eb.jpg",
       "train/7545/2685fc115f369749.jpg",
       "train/7532/f38b800f8eb54351.jpg",
       "train/6572/7889ccdf56de1679.jpg",
       "train/7551/a50b7a8977033cfd.jpg",
       "train/6572/b781facd601293c5.jpg",
       "train/7505/a3c859eb75d6e914.jpg",
       "train/7492/ff907236c99bfdfd.jpg",
       "train/7505/0765cfce36357d56.jpg",
       "train/7503/fd12ea0e07da1e3c.jpg",
       "train/7545/6f4e52b809150f40.jpg",
       "train/7532/c7a574dea7fb0623.jpg",
       "train/7551/4da4fc97b5b81c22.jpg",
       "train/7503/20c7f7f50f101cfc.jpg",
       "train/6574/cb7ed2d6d557d69a.jpg",
       "train/7505/8a6202a7a2fbdbfd.jpg",
       "train/7545/ff3deb268dc9f26d.jpg",
       "train/6572/81f1c4f662e69a0a.jpg",
       "train/7545/3c871a10cc406563.jpg"
       ]

def label_dictionary(name):
    id_lbl_dict = {}
    counter = 0
    with open(name) as file:
        for line in file:
            line = line.strip().split(",")
            image_id = line[0]
            lbls = line[1:]
            counter += 1
            print image_id, counter
            id_lbl_dict[image_id] = lbls

    return id_lbl_dict

def make_all_file(id_lbl_dict, file_dirs_file, newfile,labelsfile):
    '''
    :param id_lbl_dict: image_id to lbls
    :param file_dirs_file: file with all images in
    :param newfile:  name of new file
    :param labelsfile: label_id to labels
    :return:
    '''
    lbl_id_to_lbl = {}
    with open(labelsfile) as reader:
        for line in reader:
            lbl = line.split(",")[1].strip().replace(" ", "_")
            lbl_id = line.split(",")[0].strip()
            if lbl.startswith('"') and lbl.endswith('"'):
                lbl = lbl[1:-1]
            if lbl_id.startswith('"') and lbl_id.endswith('"'):
                lbl_id = lbl_id[1:-1]
            lbl_id_to_lbl[lbl_id] = lbl
    with open(file_dirs_file) as file, open(newfile, "w") as new:
        counter = 0
        for path in file:
            path = path.rstrip()
            array = path.split("/")
            picture_id = array[2].split(".")[0]
            print "picture_id", picture_id
            try:

                lbl_ids = id_lbl_dict[picture_id]

                new.write(path + "," + ",".join([lbl_id_to_lbl[x] for x in lbl_ids]) + "\n")

            except:
                counter += 1
                print counter

def get_classes(filename):
    '''
    :param filename: filename for train or test
    :return: class_dict with classes
    '''
    class_dict = {}
    with open(filename) as file:
        for line in file:
            the_class = line.strip()
            class_dict[the_class] = 1
    return class_dict

def make_zs_split(all_file, train_classes, test_classes, test_file, train_file):
    '''
    :param all_file: file with all images as well as lbls for each image
    :param train_classes: train classes for zs split
    :param test_classes:  test classes for zs split
    :param test_file:   placement and name of test file
    :param train_file:  placement and name of train file
    :return: None. Just make test and train file
    '''
    with open(all_file) as file, open(test_file, "w") as test, open(train_file, "w") as train:
        counter = 0
        for line in file:
            array = line.strip().split(",")
            path = array[0]
            lbls = array[1:]
            legal_lbls = []
            # make array containing lbls in the split
            # this split can be made from i.e glove_wiki_300D
            train_legal = []
            test_legal = []
            for lbl in lbls:
                if lbl in train_classes or lbl in test_classes:
                    legal_lbls.append(lbl)
                if lbl in train_classes:
                    train_legal.append(lbl)
                if lbl in test_classes:
                    test_legal.append(lbl)

            if len(legal_lbls) == 0:
                continue
            for lbl in legal_lbls:
                if lbl in test_classes:
                    test.write(path + "," + ",".join(test_legal) + "\n")
                    break
                if lbl in train_classes:
                    train.write(path + "," + ",".join(train_legal) + "\n")
                    break
            print counter, "/", "5280911"
            counter += 1
from glob import glob
import os.path as osp

def val_imgs_list(placement):
    '''
    :param placement: placement of image txt file
    :return: None. Just make file with all images in.
    '''
    with open(osp.join(placement, "validation_imgs.txt"), "w") as val:
        counter = 0
        folder_list_total = glob(osp.join(placement,"val_imgs","*"))
        for folder in folder_list_total:
            total = osp.join(folder, "*")
            file_placement = glob(total)
            for file in file_placement:
                file = file.strip().split("openimages/")[1]
                val.write(file + "\n")
                counter += 1
                print counter

def val_id_to_dir(file):
    '''
    :param file: validation file with all files
    :return: image_id to dir of file. Dictionary
    '''
    id_to_dir ={}
    counter = 0
    with open(file) as file:
        for line in file:

            line = line.strip()
            image_id = line.split("/")[2].split(".")[0]
            counter +=1
            id_to_dir[image_id] = line
    return id_to_dir



def make_val_openimages(val_classes,labelsfile, validation_file, newfile, id_to_dir):
    '''
    :param val_classes: Same as test classes
    :param labelsfile: id_to_label file
    :param validation_file: file that says labels for each image_id
    :param newfile: name and placement of validation file
    :param id_to_dir: image_id to dir of files dictionary
    :return:  Nothing. Just makes file
    '''
    lbl_id_to_lbl = {}
    with open(labelsfile) as reader:
        for line in reader:
            lbl = line.split(",")[1].strip().replace(" ", "_")
            lbl_id = line.split(",")[0].strip()
            if lbl.startswith('"') and lbl.endswith('"'):
                lbl = lbl[1:-1]
            if lbl_id.startswith('"') and lbl_id.endswith('"'):
                lbl_id = lbl_id[1:-1]
            lbl_id_to_lbl[lbl_id] = lbl

    image_id_labels = {}
    with open(validation_file) as file:
        file.next()
        for line in file:
            line = line.strip().split(",")
            image_id = line[0]
            label_id = line[2]
            label = lbl_id_to_lbl[label_id]
            if image_id not in image_id_labels:
                image_id_labels[image_id] = []
            if label in val_classes:
                image_id_labels[image_id].append(label)

    counter = 0
    with open(newfile, "w") as file:
        for image_id, lbls in image_id_labels.iteritems():
            try:
                dir = id_to_dir[image_id]
                if len(lbls) >=1:
                    file.write(dir + "," + ",".join(lbls) + "\n")
            except:
                counter +=1
                print counter







if __name__ == "__main__":
    id_lbl_file ="../../data/image_data/openimages/label_pr_picture_100.txt"
    file_dirs_file ="../../data/image_data/openimages/filedirs.txt"
    newfile = "../../data/image_data/openimages/path_lbls.txt"
    labelsfile = "../../data/image_data/openimages/labels.csv"



    #id_lbl_dict = label_dictionary(id_lbl_file)
    #make_all_file(id_lbl_dict, file_dirs_file, newfile, labelsfile)
    train_file = "../../data/image_data/openimages/train_zs.txt"
    test_file = "../../data/image_data/openimages/test_zs.txt"

    train_classes = "../../data/image_data/openimages/oi_train_classes.txt"
    test_classes = "../../data/image_data/openimages/oi_test_classes.txt"
    train_classes = get_classes(train_classes)
    test_classes = get_classes(test_classes)

    validation_file = "../../data/image_data/openimages/labels_val.csv"
    new_val_file = "../../data/image_data/openimages/validation.txt"
    #make_zs_split(newfile, train_classes, test_classes, test_file, train_file)
    #val_imgs_list(osp.join("..","..","data","image_data","openimages"))
    id_to_dir = val_id_to_dir(osp.join("..","..","data","image_data","openimages","validation_imgs.txt"))
    make_val_openimages(test_classes,labelsfile, validation_file, new_val_file, id_to_dir)







