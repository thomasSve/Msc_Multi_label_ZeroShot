import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from operator import itemgetter
from collections import Counter
from language_models.language_factory import get_language_model
from bt_datasets.factory import get_imdb
from sklearn.metrics.pairwise import cosine_distances 

def vis(x, y, xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, 'ro')
    plt.show()

def vis_distance(lang_db, imdb, seen_labels):
    # Precision results
    evaluation_path = osp.join('output', 'evaluate_results')

    with open(osp.join(evaluation_path, 'results_pr_class.txt')) as f:
        scores = []
        distances = []
        for line in f:
            words = line.strip().split(' ')
            words = [x.strip() for x in words]
            c = words[0]
            c_vector = [lang_db.word_vector(c)]
            closest_labels = lang_db.closest_labels(c_vector, k_closest = 5)
            
            closest_seen = None
            for closest in closest_labels:
                if closest in seen_labels:
                    closest_seen = closest
                    break
            if closest_seen is None:
                print "closest is none"

            cv = [np.array(lang_db.word_vector(closest_seen))]
            distances.append(float(cosine_distances(c_vector, cv)))
            scores.append(words[1])
        #print distances
        #print scores
        vis(distances, scores, 'distance', 'score')

def vis_top_classes(lang_db, imdb, seen_labels):
    evaluation_path = osp.join('output', 'evaluate_results')

    c_vector = [lang_db.word_vector('sunglass')]
    closest_labels = lang_db.closest_labels(c_vector, k_closest = 10)
    print closest_labels
    with open(osp.join(evaluation_path, 'results_pr_class.txt')) as f:
        classes = []
        for line in f:
            c = {}
            words = line.strip().split(' ')
            words = [x.strip() for x in words]
            c['label'] = words[0]
            c['score'] = words[1]
            c_vector = [lang_db.word_vector(c['label'])]
            closest_labels = lang_db.closest_labels(c_vector, k_closest = 10)
            c_labels = closest_labels
            '''
            c_labels = []
            i = 0
            for closest in closest_labels:
                if closest in seen_labels:
                    c_labels.append(closest)
                    i += 1
                    if i == 5: break
            '''     
            c['nn'] = c_labels
            classes.append(c)
        sorted_classes = sorted(classes, key=lambda k: k['score'], reverse=True)
        for c in sorted_classes[:15]:
            print "Class: {}, Score: {}, Closest: {}".format(c['label'], c['score'], c['nn'])

def find_distribution_accuracy(scores):
    dist = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 85.0, 90.0, 100.0]
    current = 0.0
    i = 0
    total = 0
    print scores
    for s in scores:
        if s > dist[i]: 
            print "{}: {}".format(dist[i], total)
            total = 1
            i += 1
            continue
        total += 1
    print "{}: {}".format(dist[i], total)
            

def vis_bar_top():
    evaluation_path = osp.join('output', 'evaluate_results')
    with open(osp.join(evaluation_path, 'imagenet_sl_results_pr_class.txt')) as f:
        scores = []
        for line in f:
            c = {}
            words = line.strip().split(' ')
            words = [x.strip() for x in words]
            scores.append(float(words[1]))
         
        sort = sorted(scores, key=float)
        find_distribution_accuracy(sort)
        x = range(len(sort))
        width = 1
        
        plt.bar(x, sort, width)
        plt.show()

def vis_graph(lang_name, imdb_name, vis):
    if 0:
        vis_bar_top()

    if 1:
        
        lang_db = get_language_model(lang_name)
        imdb = get_imdb(imdb_name)
        unseen_labels =  imdb.get_labels(1)
        unseen_seen_labels = imdb.get_labels(2)
        seen_labels = []
        for s in unseen_seen_labels:
            if s in unseen_labels:
                continue
            seen_labels.append(s)
    
        lang_db.build_tree(2, unseen_seen_labels, imdb_name)
    
        #vis_distance(lang_db, imdb, seen_labels)
        vis_top_classes(lang_db, imdb, seen_labels)
