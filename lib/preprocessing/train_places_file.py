#!/usr/bin/env python
# Made by Bjoernar Remmen
# Short is better

import os.path as osp
root = osp.join("..","..")

places = osp.join(root, "data", "image_data","places205")

image_path = osp.join("data", "vision", "torralba", "deeplearning", "images256")
from glob import glob
def get_paths():
    path = osp.join(places,image_path)
    paths = glob(osp.join(path,"*"))
    #print paths
    all_image_paths = []
    for path in paths:
        labels = glob(osp.join(path,"*"))
        for image in labels:
            images = glob(osp.join(image,"*"))
            for img in images:
                p = img
                label = p.split("/")[-2]
                all_image_paths.append((p,label))
    return all_image_paths







if __name__ == "__main__":
    tuple_list = get_paths()
    text_file = osp.join(places,"places_train.txt")
    with open(text_file,"w") as save:
        counter = 0
        for item in tuple_list:
            print counter, "/" , len(tuple_list)
            path = item[0]
            label = item[1]
            save.write(path + "," + label + "\n")
            counter += 1
