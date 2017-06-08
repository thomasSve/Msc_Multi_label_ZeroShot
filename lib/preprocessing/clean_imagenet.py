#!/usr/bin/env python
import os.path as osp
import sys

def generate_imagenet_single(lang_db, imdb):
    with open('val_bts.txt', 'w+') as wf:
        skipped = 0
        for i in range(imdb.num_images):
            if i % 1000 == 0: print "{} / {}, skipped {}".format(i, imdb.num_images, skipped)
            if imdb._image_folder() is 'val':
                y = imdb.get_val_label(i, lang_db)
            else:
                y = imdb.get_image_label(i, lang_db)
            if y is False:
                skipped = skipped + 1
                continue
            wf.write('%s' % y[0])
            wf.write(' %s\n' % y[1])

def generate_splitted_classes(lang_db, fn = 'test_zsl_bts.txt'):
    train_split_f = 'train_800_classes.txt'
    test_split_f = 'test_200_classes.txt'
    with open(fn) as f, open('map_clsloc.txt') as synset,\
         open(train_split_f, 'w+') as train_f, open(test_split_f, 'w+') as test_f:
        classes = []
        for line in synset:
            w = line.strip().split(" ")
            classes.append(w[2].lower())
            
        test_cls = []
        for line in f:
            words = line.strip().split(' ')
            cls = words[1].lower()
            if cls not in test_cls:
                test_f.write('%s\n' % cls)
                test_cls.append(cls)
                classes.remove(cls)
        print "Total test classes: {}".format(len(test_cls))

        train_cls = []
        for c in classes:
            if lang_db.word_vector(c) is not None:
                train_cls.append(c)
                train_f.write('%s\n' % c)
            
        print "Total train classes: {}".format(len(train_cls))

if __name__ == '__main__':
    
    this_dir = osp.dirname(__file__)
    lib_path = osp.join(this_dir, '../..', 'lib')
    sys.path.insert(0, lib_path)

    from bt_datasets.factory import get_imdb
    from language_models.language_factory import get_language_model

    lang_db = get_language_model('glove_wiki_300D')
    imdb = get_imdb('imagenet1k_val')
    generate_imagenet_single(lang_db, imdb)
    
    #generate_splitted_classes(lang_db)
    
