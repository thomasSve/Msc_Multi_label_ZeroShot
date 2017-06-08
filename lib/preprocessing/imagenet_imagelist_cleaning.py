import os, sys
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np

import random
import os.path as osp

def generate_no_extra(imagelist_path, name = "train_no_extra.txt", origin = 'train.txt', size = 1000):
    name = os.path.join(imagelist_path, name)
    origin = os.path.join(imagelist_path, origin)
    content = []
    with open(origin) as f, open(name, 'w') as wf:
        content = [x.strip() for x in f.readlines()]
        random.shuffle(content)
        for item in content:
            if 'extra' in item: continue
            wf.write("%s\n" % item)
    
def generate_small_train(name, origin, size):
    content = []
    with open(origin) as f, open(name, 'w') as wf:
        content = [x.strip() for x in f.readlines()]
        random.shuffle(content)
        for item in content[:size]:
            wf.write("%s\n" % item)
    print("Successfully generated small dataset ", name) 

def check_image_size(index, filename, num_classes = 201):
    tree = ET.parse(filename)
    objs = tree.findall('object')
    im_size = tree.find("size")
    im_width = float(im_size.find('width').text)
    im_height = float(im_size.find('height').text)
    im_ratio = float(im_width / im_height)
    #print im_ratio
    if not(0.117 < im_ratio < 15.5):
        #print("Image ignored due to im_ratio out of boundaries, ", index)
        return False
    if (im_width < 127 and im_height < 96) or (im_width>500 and im_height>500):
        return False        

    return True

def clean_imagelist(imagenet_path):
    added = 0
    with open(imagenet_path + '/data/imagenet/data/ImageSets/DET/train_no_extra.txt') as f, open(imagenet_path + 'clean_imagelist.txt', 'w') as wf:
        i = 0
        for line in f:
            index = line.split()
            filename = os.path.join(imagenet_path, 'ILSVRC/Annotations/DET/train', index[0] + '.xml')
            if check_image_size(index, filename):
                added = added + 1
		wf.write("%s\n" % line.strip())
                i = i+1
                if i%1000==0:
                    print added,'/',i

def random_line(filename):
    line_num = 0
    selected_line = ''
    with open(filename) as f:
        while 1:
            line = f.readline()
            if not line: break
            line_num += 1
            if random.uniform(0, line_num) < 1:
                selected_line = line
    return selected_line.strip()


def combine_synsets():
    line1 = random_line(cfg.ROOT_DIR + "/data/imagenet/data/ImageSets/DET/train_no_extra.txt")
    synset1 = line1.split("/")[1]
    line2 = random_line(cfg.ROOT_DIR + "/data/imagenet/data/ImageSets/DET/train_no_extra.txt")
    synset2 = line2.split("/")[1]

    variable = 0
    if synset1 == synset2:
        variable =1

    return line1, line2, variable


def combine_synsets2(eq=None):
    line1 = random_line(cfg.ROOT_DIR + "/data/imagenet/data/ImageSets/DET/train_no_extra.txt")
    synset1 = line1.split("/")[1]
    get_new = True
    while get_new:
        line2 = random_line(cfg.ROOT_DIR + "/data/imagenet/data/ImageSets/DET/train_no_extra.txt")
        synset2 = line2.split("/")[1]

        if synset1 == synset2:
            variable =1
        else:
            variable = 0
        if variable == eq:
            get_new = False
            return line1, line2, variable


if __name__ == "__main__":
    sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
    from config import cfg

    with open(cfg.ROOT_DIR + "/data/imagenet/data/ImageSets/DET/train_siamese.txt", "w") as save:
        for i in xrange(100):
            eq = 0
            if i % 2 == 0:
                eq = 1
            line1, line2, variable =combine_synsets2(eq=eq)
            save.write(line1 + " " + line2 + " " + str(variable))

    #root_path = cfg.ROOT_DIR
    #clean_imagelist(root_path)


    
    #imagelist_path = os.path.join(imagenet_path, "ILSVRC/ImageSets/DET")
    #generate_no_extra(imagelist_path)
