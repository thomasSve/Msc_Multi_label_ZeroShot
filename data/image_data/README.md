Folder to populate image data for training and testing, similar structure as py-faster-rcnn:

```
ILSVRC13 
└─── Images
    │   *.JPEG (Image files, ex:ILSVRC2013_val_00000565.JPEG)
└─── Annotations
    |   *.xml
└─── ImageSets
    │   train.txt
└─── Generated_proposals
    │   vgg_cnn_m_1024_rpn_stage1_iter_90000_proposals.pkl
```

Follow [this](https://github.com/deboc/py-faster-rcnn/tree/master/help) documentation on how to train on own dataset on [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). Remember when training with on zero shot to add a imageset that contains the untrained classes.
