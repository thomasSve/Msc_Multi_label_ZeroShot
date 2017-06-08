import os, cv2, operator
import os.path as osp
from keras.models import load_model
from bt_net.generate_boxes import generate_n_boxes
from bt_net.train_bts import euclidean_distance
from bt_net.faster_rcnn_boxes import load_frcnn_boxes, vis_proposals, boxes_to_images, load_all_boxes_yolo, load_image_boxes_yolo
from preprocessing.preprocess_images import read_img
import numpy as np
from multiprocessing import Pool


def predict_image_singlelabel(img_path, model, lang_db, extra = None):
    img = read_img(img_path, target_size=(299, 299))
    img = np.expand_dims(img, axis=0)
    pred_vector = model.predict(img)
    predictions = lang_db.closest_labels(pred_vector, k_closest = 20)
    return predictions

def predict_ml_frcnn(img_path, model, lang_db, extra, k_closest = 1):
    img = read_img(img_path)
    boxes = boxes_to_images(img_path, img, extra)
    pred_vects = model.predict_on_batch(boxes) # Return predictions on the boxes, predictions 300d vector
    
    predictions = lang_db.closest_labels(pred_vects, k_closest=k_closest) # Return closest label to each vector box
    dict_counter = {}
    preds = []
    for k in range(k_closest):
        for i in range(len(predictions)):
            preds.append(predictions[i][k])
        
    for pred in preds:
        if pred in dict_counter:
            dict_counter[pred] += 1
        else:
            dict_counter[pred] = 1
    sorted_dict = sorted(dict_counter.items(), key=operator.itemgetter(1), reverse=True)
    return [x[0] for x in sorted_dict[:25]]


def predict_ml_random(img_path, model, lang_db, extra):
    # Generate boxes for image
    img = read_img(img_path, target_size=(299, 299))
    boxes = generate_n_boxes(img, pool = extra)
    pred_vects = model.predict_on_batch(boxes) # Return predictions on the boxes, predictions 300d vector
    predictions = lang_db.closest_labels(pred_vects, k_closest=1) # Return closest label to each vector box
    dict_counter = {}
    for pred in predictions:
        pred = pred[0]
        # Retrieve closest word in semantic space
        if pred in dict_counter:
            dict_counter[pred] += 1
        else:
            dict_counter[pred] = 1
    
    sorted_dict = sorted(dict_counter.items(), key=operator.itemgetter(1), reverse=True)
    return [x[0] for x in sorted_dict[:20]]

def predict_ml_yolo(img_path, model, lang_db, dictionary,k_closest = 1):
    img = read_img(img_path)
    boxes = load_image_boxes_yolo(img, dictionary, img_path)
    if len(boxes) == 0: return []
    pred_vects = model.predict_on_batch(boxes)  # Return predictions on the boxes, predictions 300d vector

    predictions = lang_db.closest_labels(pred_vects, k_closest=k_closest)  # Return closest label to each vector box
    dict_counter = {}
    preds = []

    for k in range(k_closest):
        for i in range(len(predictions)):
            preds.append(predictions[i][k])

    for pred in preds:
        if pred in dict_counter:
            dict_counter[pred] += 1
        else:
            dict_counter[pred] = 1
    sorted_dict = sorted(dict_counter.items(), key=operator.itemgetter(1), reverse=True)
    return [x[0] for x in sorted_dict[:25]]

def test_bts(lang_db, imdb, checkpoint, euc_loss, singlelabel_predict, space, box_method):
    # Load trained model
    if euc_loss:
        print "Detected euclidean loss"
        model = load_model(checkpoint, custom_objects={'euclidean_distance': euclidean_distance})
        model_loss = "euclidean"
    else:
        model = load_model(checkpoint)
        model_loss = str(model.loss)

    index = [ix for ix in range(imdb.num_images)]
    output_path = osp.join('output', 'predict_results')
    if not osp.exists(output_path):
        os.makedirs(output_path)

    if singlelabel_predict:
        predict_image = predict_image_singlelabel
        pred = '_sl_'
    elif box_method == 'yolo':
        predict_image = predict_ml_yolo
        '''
        dictionary
        '''
        extra = load_all_boxes_yolo(imdb)
        pred = '_ml_yolo_'
    elif box_method == 'frcnn':
        predict_image = predict_ml_frcnn
        boxes = load_frcnn_boxes(imdb)
        pred = '_ml_frcnn_'
    else:
        predict_image = predict_ml_random
        pred = '_ml_random'
        extra = Pool(4)
        
    output_file = osp.join(output_path, 'results_'  + imdb.name + '_' + str(lang_db.name) + pred + str(model_loss) + '_' + str(space) + '_l2_nw.txt')
    with open(output_file, 'w+') as wf:
        for i in index:
            if i%100==0: print "Generating proposals: {}/{}".format(i, imdb.num_images)
            if box_method == 'frcnn': extra = boxes[i]
            img_path = imdb.image_path_at(i)
            predictions = predict_image(img_path, model, lang_db, extra)

            if box_method == 'yolo':
                if len(predictions) == 0: continue
            # write predictions to file
            # id, num_actual, a1, a2, ..., an, num_predicted, p1, p2, ..., pn
            actual = imdb.gt_at(i)
            wf.write('%s' % imdb.image_index_path(i))
            if isinstance(actual, basestring): # Check if gt is a simple string and not array
                wf.write(', %s' % 1) # Length of ground_truth is 1
                wf.write(', %s' % actual)
            else:
                wf.write(', %s' % len(actual))
                for a in actual:
                    wf.write(', %s' % a)
            
            wf.write(', %s' % len(predictions))
            if len(predictions) > 1:
                for p in predictions:
                    wf.write(', %s' % p)
            else:
                wf.write(', %s' % predictions)
                
            wf.write('\n')

