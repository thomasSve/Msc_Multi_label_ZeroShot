#!/usr/bin/env python

import os, sys
import os.path as osp
from itertools import izip


def clean_backslashes(imagelist, clean_list):
    # Change backslashes to forwardslashes
    with open(imagelist) as f, open(clean_list, 'w') as wf:
        for line in f:
            words = line.split('\\')
            wf.write('%s' % words[0])
            wf.write('/%s\n' % words[1].strip())


def remove_no_tags(imagelist, clean_list, tag_list, clean_tag_list):
    # Read line for line in taglist, if vector doesnt contain a 1 (sum of vector is 0), do not add to list, if sum > 0, add image to imagelist, and vector to clean taglist

    with open(imagelist) as ilf, \
            open(clean_list, 'w') as clf, \
            open(tag_list) as tlf, \
            open(clean_tag_list, 'w') as ctlf:

        tot_img = 0
        clean_img = 0
        for tag_line, image_line in izip(tlf, ilf):
            tot_img = tot_img + 1
            numbers = tag_line.strip().split()

            numbers = [int(x) for x in numbers]
            if sum(numbers) > 0:
                ctlf.write(tag_line)
                clf.write(image_line)
                clean_img = clean_img + 1

        print "Finished with total images: {} of total {}".format(clean_img, tot_img)


def make_925_train(nwh):
    labels_81 = open(osp.join(nwh, "Concepts", "Concepts81.txt")).readlines()
    labels_1k = open(osp.join(nwh, "Concepts", "TagList1k.txt")).readlines()

    label_index = []
    for i in range(len(labels_81)):
        for j in range(len(labels_1k)):
            if labels_81[i].rstrip().lower() == labels_1k[j].rstrip().lower():
                label_index.append(j)

    return label_index


def check_overlap(indices, numbers):
    for ind in indices:
        if numbers[ind] == 1:
            return True


def remove_overlap(nwh, imagelist, clean_list, tag_list, clean_tag_list, test_zs_u_im, test_us_im):
    # ImageList_1k
    # ImageList 81k
    # Write labels from original im_list at that same place. Labels 81 and 1k

    overlapping = make_925_train(nwh)
    with open(imagelist) as im_list, \
            open(clean_list, 'w') as cl_im_list, \
            open(tag_list) as tag_list, \
            open(clean_tag_list, 'w') as cl_tg_list:
        tot_img = 0
        clean_img = 0
        for tag_line, image_line in izip(tag_list, im_list):
            tot_img += 1
            numbers = tag_line.strip().split()
            numbers = [int(x) for x in numbers]
            if check_overlap(overlapping, numbers):
                continue
            else:
                cl_im_list.write(image_line)
                cl_tg_list.write(tag_line)
                clean_img += 1
    print "Finished with total images: {} of total {}".format(clean_img, tot_img)
    orginal_1k_tag = open(osp.join(nwh, "Train_Tags1k.txt")).readlines()
    original_81_tag = open(osp.join(nwh, "Train_Tags81.txt")).readlines()
    labels_81 = open(osp.join(nwh, "Concepts", "Concepts81.txt")).readlines()
    labels_1k = open(osp.join(nwh, "Concepts", "TagList1k.txt")).readlines()
    original_imagelist = open(osp.join(nwh, "ImageList", "TrainImagelist.txt")).readlines()
    seen_unseen_counter = 0
    unseen_counter = 0
    with open(test_zs_u_im, "w") as test_zs_u_img, \
            open(test_us_im, "w") as test_zs_us_img:
        imagelist1k_zs = open(clean_list).readlines()
        for i in range(len(original_imagelist)):
            img = original_imagelist[i]
            if img not in imagelist1k_zs:

                line_1k = orginal_1k_tag[i].rstrip().split()
                line_81 = original_81_tag[i].rstrip().split()
                vector_1k_ind = [i for i, z in enumerate([int(x) for x in line_1k]) if z != 0]
                vector_81 = [vec_81_ind for vec_81_ind, val in enumerate([int(y) for y in line_81]) if val != 0]
                lbl81 = []
                lbl1k = []
                for n in vector_1k_ind:
                    lbl1k.append(labels_1k[n].strip())
                for m in vector_81:
                    lbl81.append(labels_81[m].strip())
                seen_unseen_list = list(set(lbl81 + lbl1k))
                unseen_list = list(set(lbl81))
                if len(seen_unseen_list) != 0:
                    test_zs_us_img.write(img.strip() + "," + ",".join(seen_unseen_list) + "\n")
                    seen_unseen_counter += 1
                if len(unseen_list) != 0:
                    test_zs_u_img.write(img.strip() + "," + ",".join(unseen_list) + "\n")
                    unseen_counter += 1

    print "Finished with total u/s: {} , u: {}".format(seen_unseen_counter, unseen_counter)


def non_overlap_file(classes1, classes2, nwh, filename):
    non_overlap = {}

    for cls in classes2:
        if cls.rstrip().lower() not in non_overlap and cls not in classes1:
            non_overlap[cls.rstrip().lower()] = 1

    with open(osp.join(nwh, filename), "w") as file_wr:
        for the_class in non_overlap.keys():
            file_wr.write(the_class.rstrip().lower() + "\n")


def resulting_classes(nwh):
    class_ind = {}
    tag_file = osp.join(nwh, 'Train_Tags925.txt')
    with open(tag_file) as tags:
        for line in tags:
            numbers = line.strip().split()
            numbers = [int(x) for x in numbers]
            indices = [i for i, x in enumerate(numbers) if x != 0]
            for ind in indices:
                if ind not in class_ind:
                    class_ind[ind] = ind

    labels_1k = open(osp.join(nwh, "Concepts", "TagList1k.txt")).readlines()
    with open(osp.join(nwh, "Train_Tags" + str(len(class_ind.keys())) + "_clean"), "w") as file:
        for index in class_ind.keys():
            file.write(labels_1k[index].rstrip() + "\n")


def train_img_lbl_clean(imagelist, image_tags, labels_file, newfile):
    imagelist = open(imagelist).readlines()
    image_tags = open(image_tags).readlines()
    labels = open(labels_file).readlines()
    with open(newfile, "w") as file:
        for img, tag in izip(imagelist, image_tags):
            tag = tag.strip().split()
            indices = [i for i, x in enumerate([int(x) for x in tag]) if x != 0]
            lbls = [labels[ind].strip() for ind in indices]
            file.write(img.rstrip() + "," + ",".join(lbls) + "\n")


def clean_train_test(lang_name, train_u_s, train_u, test):
    from language_models.language_factory import get_language_model
    lang_db = get_language_model(lang_name)
    print lang_db
    train_u_s_read = open(train_u_s).readlines()
    print "len(train_us)", len(train_u_s_read)
    unique_lbl_t = {}
    unique_lbl_u = {}
    unique_lbl_us = {}

    train_u_read = open(train_u).readlines()
    print "len(train_u)", len(train_u_s_read)

    test_read = open(test).readlines()
    print "len(train)", len(train_u_s_read)


    with open(train_u_s, "w") as train_u_s_file, \
            open(train_u, "w") as train_u_file, \
            open(test, "w") as test_file:
        for img_lbls in train_u_s_read:
            line = img_lbls.split(",")

            img = line[0].strip()
            lbls = [x.strip() for x in line[1:]]
            new_lbls = []
            for lbl in lbls:
                lbl = lbl.strip()
                vec = lang_db.word_vector(lbl)
                if vec is not None:
                    new_lbls.append(lbl)
                    if lbl is not None:
                        unique_lbl_us[lbl] = 1
            img_path = img.split("\\")
            path = img_path[0] + "/" + img_path[1]
            train_u_s_file.write(path + "," + ",".join(set(new_lbls)) + "\n")
            print "s/u",path, new_lbls


        for img_lbls in train_u_read:
            line = img_lbls.split(",")
            img = line[0].strip()
            lbls = [x.strip() for x in line[1:]]
            new_lbls = []
            for lbl in lbls:
                lbl = lbl.strip()
                vec = lang_db.word_vector(lbl.strip())
                if vec is not None:
                    new_lbls.append(lbl.strip())
                    if lbl is not unique_lbl_us:
                        unique_lbl_u[lbl] = 1
            img_path = img.split("\\")
            path = img_path[0] + "/" + img_path[1]
            print "unseen",path, new_lbls
            train_u_file.write(path + "," + ",".join(set(new_lbls)) + "\n")

        for img_lbls in test_read:
            line = img_lbls.split(",")
            img = line[0].strip()
            lbls = [x.strip() for x in line[1:]]
            new_lbls = []

            for lbl in lbls:
                lbl = lbl.strip()
                vec = lang_db.word_vector(lbl)
                if vec is not None:
                    new_lbls.append(lbl)
                    if lbl is not unique_lbl_us:
                        unique_lbl_t[lbl] = 1

            img_path = img.split("\\")
            path = img_path[0] + "/" + img_path[1]
            print "test",path, new_lbls

            test_file.write(path + "," + ",".join(set(new_lbls)) + "\n")

def train_1k(tags, imagelist, newfilename):
    orginal_1k_labels = open(osp.join(nwh, "Concepts", "TagList1k.txt")).readlines()
    with open(tags) as tag, open(imagelist) as img_l, open(newfilename, "w") as newfile:
        for line in tag:
            line = line.rstrip().split()
            numbers = [int(x) for x in line]
            indices = [i for i,x in enumerate(numbers) if x != 0]
            print indices
            labels = []
            for ind in indices:
                labels.append(orginal_1k_labels[ind].rstrip())
            img_dir = img_l.next()
            newfile.write(img_dir.rstrip() + "," + ",".join(labels)+ "\n")







if __name__ == '__main__':
    nwh = '../../data/image_data/nus_wide'
    '''
    # clean_backslashes('train.txt', 'train_clean.txt')
       remove_no_tags(osp.join(nwh, 'ImageList', 'TrainImagelist.txt'),
                   osp.join(nwh, 'ImageList', 'TrainImagelist_clean_81.txt'),
                   osp.join(nwh, 'Train_Tags81.txt'),
                   osp.join(nwh, 'Train_Tags81_clean.txt')
                   )

    remove_no_tags(osp.join(nwh, 'ImageList', 'TestImagelist.txt'),
                   osp.join(nwh, 'ImageList', 'TestImagelist_clean.txt'),
                   osp.join(nwh, 'Test_Tags81.txt'),
                   osp.join(nwh, 'Test_Tags81_clean.txt')
                   )

    remove_no_tags(osp.join(nwh, 'ImageList', 'TestImagelist.txt'),
                   osp.join(nwh, 'ImageList', 'TestImagelist_clean.txt'),
                   osp.join(nwh, 'Test_Tags1k.txt'),
                   osp.join(nwh, 'Test_Tags1k_clean.txt')
                   )
    remove_no_tags(osp.join(nwh, 'ImageList', 'TestImagelist.txt'),
                   osp.join(nwh, 'ImageList', 'TestImagelist_clean.txt'),
                   osp.join(nwh, 'Train_Tags1k.txt'),
                   osp.join(nwh, 'Train_Tags1k_clean.txt')
                   )

    clean_backslashes(osp.join(nwh, 'ImageList', 'TrainImagelist_clean.txt'),
                      osp.join(nwh, 'train_clean.txt')
                      )
    clean_backslashes(osp.join(nwh, 'ImageList', 'TestImagelist_clean.txt'),
                      osp.join(nwh, 'test_clean.txt')
                      )

'   '''
    '''
    remove_no_tags(osp.join(nwh, 'ImageList', 'TrainImagelist.txt'),
                   osp.join(nwh, 'ImageList', 'TrainImagelist_clean_81.txt'),
                   osp.join(nwh, 'Train_Tags81.txt'),
                   osp.join(nwh, 'Train_Tags81_clean.txt')
                   )
    remove_no_tags(osp.join(nwh, 'ImageList', 'TrainImagelist.txt'),
                   osp.join(nwh, 'ImageList', 'TrainImagelist_clean_1k.txt'),
                   osp.join(nwh, 'Train_Tags1k.txt'),
                   osp.join(nwh, 'Train_Tags1k_clean.txt')
                   )


    remove_overlap(nwh, \
                   osp.join(nwh, 'ImageList', 'TrainImagelist_clean_1k.txt') ,\
                   osp.join(nwh, 'ImageList', 'TrainImagelist_925.txt'), \
                   osp.join(nwh, 'Train_Tags1k_clean.txt'), \
                   osp.join(nwh, 'Train_Tags925.txt'),\
                   osp.join(nwh, 'ImageList', 'Test_zs_u_img_lbl.txt'), \
                   osp.join(nwh, "ImageList", "Test_zs_us_img_lbl.txt"))

    train_img_lbl_clean(osp.join(nwh, 'ImageList', 'TrainImagelist_clean_1k.txt'), \
                                 osp.join(nwh, 'Train_Tags1k_clean.txt'),\
                                 osp.join(nwh, "Concepts", "TagList1k.txt"), \
                                 osp.join(nwh,"ImageList", "Train_zs_920_img_lbl.txt"))
        '''
    '''
    this_dir = osp.dirname(__file__)
    lib_path = osp.join(this_dir, '..', '..', 'lib')
    sys.path.insert(0, lib_path)
    nwh = osp.join(this_dir, "..", "..", "data", "image_data", "nus_wide")
    import os

    print("abspath", os.path.abspath(nwh))
    print os.path.exists(osp.join(nwh, 'ImageList', 'Test_zs_u_img_lbl.txt'))
    print os.path.exists(osp.join(nwh, "ImageList", "Test_zs_us_img_lbl.txt"))
    print os.path.exists(osp.join(nwh, 'ImageList', 'Test_zs_u_img_lbl.txt'))

    clean_train_test('glove_wiki_300', osp.join(nwh, 'ImageList', 'Test_zs_us_img_lbl.txt'),
                     osp.join(nwh, "ImageList", "Test_zs_u_img_lbl.txt"), \
                     osp.join(nwh, "ImageList", "Train_zs_920_img_lbl.txt"))
    '''
    # labels_81 = open(osp.join(nwh, "Concepts", "Concepts81.txt")).readlines()
    # labels_1k= open(osp.join(nwh, "Concepts", "TagList1k.txt")).readlines()
    # non_overlap_file(labels_81, labels_1k, nwh, "Train_925_classes.txt")
    # resulting_classes(nwh)
    # create_test(nwh,  osp.join(nwh, 'ImageList', 'TrainImagelist_clean.txt'),\
    #            osp.join(nwh, 'ImageList', 'TrainImagelist_925.txt'), \
    #            osp.join(nwh, 'Train_Tags81_clean.txt'), \
    #            osp.join(nwh, 'ImageList', 'Test_imagelist_zs.txt'),  osp.join(nwh, 'Test_tags_all_zs.txt'),
    #            osp.join(nwh, 'Test_tags_unseen.txt'))

    clean_backslashes(osp.join(nwh,"ImageList", "TrainImagelist_clean_1k.txt"), osp.join(nwh,"TrainImagelist_bs_clean_1k.txt"))
    train_1k(osp.join(nwh,"Train_Tags1k_clean.txt"), osp.join(nwh, "TrainImagelist_bs_clean_1k.txt"), osp.join(nwh, "train_1k.txt"))