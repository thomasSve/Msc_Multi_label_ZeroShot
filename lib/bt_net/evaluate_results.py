# 
# File to evaluate results
# 
# Per-Class recall, per-class precision, overall-recall,
# overall-precision, percentage of recalled labels in all labels (N+)
# mean-average-precision (map)
# 

import numpy as np
import os.path as osp
import operator
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from annoy import AnnoyIndex
import os


def overall_precision(correctly_annotated, n_preds):
    """ Overall Precision """ 
    n_c = 0
    n_p = 0
    for key, value in correctly_annotated.iteritems():
        n_c += value
        n_p += n_preds[key]
        
    return round(float(n_c) / n_p, 5)
    
   
def per_class_precision(correctly_annotated, n_preds):
    """ Per-Class prediction """
    precision = 0.0
    for key, value in correctly_annotated.iteritems():
        if value != 0 and n_preds[key] != 0:
            precision += float(value) / n_preds[key]
            
    return round(precision / float(len(correctly_annotated)), 5)

def overall_recall(correctly_annotated, gt):
    """ Overall Recall """
    n_c = 0
    n_g = 0
    for key, value in correctly_annotated.iteritems():
        n_c += value
        n_g += gt[key]
        
    return round(float(n_c) / n_g, 5)

def per_class_recall(correctly_annotated, gt):
    """ Per-class Recall """
    recall = 0
    for key, value in correctly_annotated.iteritems():
        if value != 0 and gt[key] != 0:
            recall += float(value)/gt[key]
        
    return round(recall / float(len(correctly_annotated)), 5)

def apk(actual, predicted, k):
    """ Average precision at k """

    if len(predicted)>k:
        predicted = predicted[:k]
    
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return round(score / min(len(actual), k), 5)

def mapk(actual, predicted, k):
    """ 
    Computes the mean average precision at k 
    """
    return round(np.mean([apk(a,p,k) for a,p in zip(actual, predicted)]), 4) * 100

def flat_hit(actual, predicted, k):
    corr = 0
    for a, p in zip(actual, predicted):
        if a[0] in p[:k]:
            corr += 1

    return round(corr / float(len(actual)), 4) * 100 # return percentage

def total_cosine(total_actual, predicted, singlelabel,lang_mod, weighted=False):
    total = 0
    counter = 0
    if weighted:
        count_dict, largest = total_dict(total_actual)
    if singlelabel:
        for gt, pred in zip(total_actual, predicted):
            if weighted:
                count = count_dict[gt[0].lower()]
                total += largest/count * cosine_similarity([lang_mod.word_vector(gt[0].lower())],
                                                                [lang_mod.word_vector(pred[0].lower())])

            else:
                count = 1
                #if gt[0].lower()!=pred[0].lower():
                counter += 1
                total += (1 / float(count)) *cosine_similarity([lang_mod.word_vector(gt[0].lower())], [lang_mod.word_vector(pred[0].lower())])
        print "Average cosine similarity - Single label"
        if weighted:
            print round(total / float(len(total_actual)),5)
        else:
            print round(total / float(counter),5)

    else:
        number_wrong_iter = 0
        total_cosine_var = 0
        number_words = 0
        for gt, pred in zip(total_actual, predicted):
            t = AnnoyIndex(300, metric="euclidean")
            for i in range(len(gt)):
                t.add_item(i, lang_mod.word_vector(gt[i].lower()))
            t.build(10)  # 10 trees
            temp_cosine = 0
            num_words_used = 0
            for pred_word in pred:
                predicted_word_vector = lang_mod.word_vector(pred_word.lower())
                indices = t.get_nns_by_vector(predicted_word_vector, 1, search_k=-1, include_distances=False)
                if gt[indices[0]] == pred_word: continue
                #else:
                #    print gt[indices[0]], pred_word
                num_words_used +=1
                temp_cosine += cosine_similarity([lang_mod.word_vector(gt[indices[0]])], [lang_mod.word_vector(pred_word)])

            if num_words_used>0:
                number_wrong_iter += 1
                temp_cosine /= float(num_words_used)
            total += temp_cosine
            for pred_word in pred:
                predicted_word_vector = lang_mod.word_vector(pred_word.lower())
                indices = t.get_nns_by_vector(predicted_word_vector, 1, search_k=-1, include_distances=False)
                total_cosine_var += cosine_similarity([lang_mod.word_vector(gt[indices[0]])], [lang_mod.word_vector(pred_word)])
                number_words += 1

        print "Average cosine similarity - Multi label"
        print "missed avg cosine", round(total / float(len(total_actual)), 5)
        print "Regular cosine", round(total_cosine_var / float(number_words), 5)
        #print round(total / float(number_wrong_iter), 5)

def total_dict(total_actual):
    # Count occurences of words, to be able to weight importance of wrong
    count_dict = {}
    largest = 0
    for list in total_actual:
        for word in list:
            if word not in count_dict: count_dict[word] = 0
            count_dict[word] += 1
    for key, value in count_dict.iteritems():
        if value > largest:
            largest = value
    return count_dict, largest

def average_cosine(fn, lang_mod):
    single_label = True
    with open(fn) as f:
        total_actual = []
        predicted = []
        line_count = 0
        print "Reading from file..."
        for line in f:
            if line_count % 10000 == 0:
                print("Number of lines evaluated: {}").format(line_count)
            # id, num_actual, a1, a2, ..., an, num_predicted, p1, p2, ..., pn
            words = line.strip().split(',')
            words = [x.strip() for x in words]
            num_actual = int(words[1])
            if single_label and num_actual != 1:
                print num_actual
                single_label = False  # Check if trying to predict multilabel

            actual = words[2:(2 + num_actual)]
            num_pred = int(words[(2 + num_actual)])
            preds = words[(3 + num_actual):]
            total_actual.append(actual)
            predicted.append(preds)
            line_count += 1

    total_cosine(total_actual, predicted, single_label, lang_mod)

def evaluate_flat_map(fn, k_values = [1, 2, 5, 10]):
    single_label = True
    with open(fn) as f:
        total_actual = []
        predicted = []
        line_count = 0
        print "Reading from file..."
        for line in f:
            if line_count % 10000 == 0:
                print("Number of lines evaluated: {}").format(line_count)
            # id, num_actual, a1, a2, ..., an, num_predicted, p1, p2, ..., pn            
            words = line.strip().split(',')
            words = [x.strip() for x in words]
            num_actual = int(words[1])
            if single_label and num_actual != 1:
                print num_actual
                single_label = False # Check if trying to predict multilabel
            
            actual = words[2:(2+num_actual)]
            num_pred = int(words[(2+num_actual)])
            preds = words[(3+num_actual):]
            total_actual.append(actual)
            predicted.append(preds)
            line_count += 1

        # Measure results
        print "Measuring results..."
        results = '_'
        for k in k_values:
            results += ' & ' + str(mapk(total_actual, predicted, k))
        if single_label:
            for k in k_values:
                results += ' & ' + str(flat_hit(total_actual, predicted, k))

        
        # Print table
        if single_label: headers = 'Map 1 2 5 10 | Flat 1 2 5 10' 
        else: headers = 'Map 1 2 5 10'
        print headers
        print results

def evaluate_results(fn, k_values = [1, 2, 5, 10]):
    # Load file with predictions
    # Calculate correctly annotated pr class
    # Get number of GT labeling for each label
    # Get number of predictions for each label
    single_label = True
    with open(fn) as f:
        image_ids = []
        total_actual = []
        predicted = []
        correct_ann = {}
        n_gt = {}
        total_preds = {}
        line_count = 0

        print "Reading from file"
        for line in f:
            if line_count % 1000 == 0:
                print("Number of lines evaluated: {}").format(line_count)
                #print("Length of corrected annotated: {}").format(len(correct_ann))
                #print("Length of gt: {}").format(len(actual))
                #print("Length of pred: {}").format(len(predicted))
            # id, num_actual, a1, a2, ..., an, num_predicted, p1, p2, ..., pn            
            words = line.strip().split(',')
            words = [x.strip() for x in words]
            num_actual = int(words[1])
            if single_label and num_actual != 1:
                print num_actual
                single_label = False # Check if trying to predict multilabel
            
            actual = words[2:(2+num_actual)]
            num_pred = int(words[(2+num_actual)])
            preds = words[(3+num_actual):]
            preds = preds[:k] # only select the k values

            image_ids.append(words[0])
            total_actual.append(actual)
            predicted.append(preds)
            # Check if correct label exists in dicts
            for a in actual:
                if a in correct_ann:
                    # If the correct label exist in predicted labels
                    if a in preds:
                        correct_ann[a] += 1
                    n_gt[a] += 1
                else:
                    correct_ann[a] = 0
                    n_gt[a] = 0
                    # Add corrected to number of pred dict
                    if a not in total_preds:
                        total_preds[a] = 0
                
            # Add predicted words to number of predictions
            for p in preds:
                if p in total_preds:
                    total_preds[p] += 1
                else:
                    total_preds[p] = 0
            line_count += 1


    
        # Measure results
        results = '_'
        results += ' & ' + str(mapk(total_actual, predicted))
        if single_label: results += ' & ' + str(flat_hit(total_actual, predicted))
        results += ' & ' + str(per_class_recall(correct_ann, n_gt))
        results += ' & ' + str(per_class_precision(correct_ann, total_preds))
        results += ' & ' + str(overall_recall(correct_ann, n_gt))
        results += ' & ' + str(overall_precision(correct_ann, total_preds))

        # Print table
        if single_label: headers = '\thead{Loss} & \thead{MAP@k} & \thead{Flat hit@k}& \thead{Per-class\\ recall} & \thead{Per-class \\ precision} & \thead{Overall\\ Recall} & \thead{Overall\\ precision}\\'
        else: headers = '\thead{Loss} & \thead{MAP@k} & \thead{Per-class\\ recall} & \thead{Per-class \\ precision} & \thead{Overall\\ Recall} & \thead{Overall\\ precision}\\'
        print headers

        
def evaluate_pr_class(fn, k = 5):
    single_label = True
    with open(fn) as f:
        line_count = 0
        correct_ann = {}
        n_gt = {}
        print "Reading from file..."
        for line in f:
            if line_count % 10000 == 0:
                print("Number of lines evaluated: {}").format(line_count)
            # id, num_actual, a1, a2, ..., an, num_predicted, p1, p2, ..., pn            
            words = line.strip().split(',')
            words = [x.strip() for x in words]
            num_actual = int(words[1])
            if single_label and num_actual != 1:
                print num_actual
                single_label = False # Check if trying to predict multilabel
            
            actual = words[2:(2+num_actual)]
            num_pred = int(words[(2+num_actual)])
            preds = words[(3+num_actual):]
            line_count += 1
            for a in actual:
                if a in correct_ann:
                    # If the correct label exist in predicted labels
                    if a in preds[:k]:
                        correct_ann[a] += 1
                    n_gt[a] += 1
                else:
                    correct_ann[a] = 0
                    n_gt[a] = 0
            
        output_path = osp.join('output', 'evaluate_results')
        if not osp.exists(output_path):
             os.makedirs(output_path)
             
        with open(osp.join(output_path, 'results_pr_class.txt'), 'w+') as wf:
            for c, gt in n_gt.iteritems():
                f_r = round(correct_ann[c] / float(gt), 4) * 100
                wf.write('%s' % c)
                wf.write(' %s\n' % f_r)

        #show_actual_predicted()

def show_actual_predicted(fn):
    single_label = True
    with open(fn) as f:
        line_count = 0
        correct_ann = {}
        n_gt = {}
        predicted = {}
        print "Reading from file..."
        for line in f:
            if line_count % 10000 == 0:
                print("Number of lines evaluated: {}").format(line_count)
            # id, num_actual, a1, a2, ..., an, num_predicted, p1, p2, ..., pn            
            words = line.strip().split(',')
            words = [x.strip() for x in words]
            actual = words[2]
            preds = words[4:]
            line_count += 1
            if actual in predicted:
                if preds[0] in predicted[actual]:
                    predicted[actual][preds[0]] += 1
                else:
                    predicted[actual][preds[0]] = 0
                # If the correct label exist in predicted labels
            else:
                predicted[actual] = {preds[0]: 1}

        output_path = osp.join('output', 'evaluate_results')
        if not osp.exists(output_path):
             os.makedirs(output_path)
             
        with open(osp.join(output_path, 'img_sl_top_predicted_pr_class.txt'), 'w+') as wf:
            for key, value in predicted.items():
                sorted_a = sorted(value.items(), key=operator.itemgetter(1), reverse=True)
                wf.write('%s' % key)
                wf.write(' %s\n' % sorted_a[:5])

def average_gt_classes(fn):
    with open(fn) as f:
        line_count = 0
        num_actual_list = []
        num_pred_list = []
        scores = []
        for line in f:
            if line_count % 10000 == 0:
                print("Number of lines evaluated: {}").format(line_count)
                
            words = line.strip().split(',')
            words = [x.strip() for x in words]
            num_actual = int(words[1])
            
            if num_actual > 50: print num_actual
            num_actual_list.append(num_actual)
            num_pred = int(words[(2+num_actual)])
            num_pred_list.append(num_pred)
            
            actual = words[2:(2+num_actual)]
            preds = words[(3+num_actual):]
            
            scores.append(apk(actual, preds, 1001))
            
            line_count += 1

        sum_actual = sum(num_actual_list)
        sum_preds = sum(num_pred_list)
        avg_pred = float(sum_preds)/line_count
        avg_actual = float(sum_actual)/line_count
        print "Average actual: ", avg_actual
        print "Average pred: ", avg_pred
        print "Mean average precision: ", round(np.mean(scores), 4) * 100
        # Calculate average score for plots:
        x, y = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(num_pred_list, scores) if xVal==a])) for xVal in set(num_pred_list)))
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        print num_actual
        plt.ylabel('Average Precision')
        plt.xlabel('Number of actual')
        plt.plot(x, y, 'r-', linewidth=2)
        plt.figtext(.8, .8, "Average actual: " + str(avg_actual))
        plt.figtext(.8, .75, "Average predicted: " + str(avg_pred))

        plt.axis([0, 25, 0, 0.5])
        #plt.show()
        
        from matplotlib2tikz import save as tikz_save
        tikz_save('test.tex')
        
