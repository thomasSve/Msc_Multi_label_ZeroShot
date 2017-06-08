#!/usr/bin/env python
"""Factory method for easily load pretrained language vector models by name """

__sets = {}

from language_models.glove_factory import glove_factory

# Set up language vectors
#for corpus in ['wiki']:
#    for dimension in [50, 150, 300]:
for corpus, dimension in [('glove_wiki_50D',50), ('glove_wiki_150D',150),\
                          ('glove_wiki_300D',300),('glove_pretrained',300),\
                          ('w2v_wiki_50D',50), ('w2v_wiki_150D',150),  ("w2v_wiki_300D",300),\
                          ('w2v_pretrained',300), ("fast_eng",300),( "fast_nor",300)]:

    name = '{}'.format(corpus)
    __sets[name] = (lambda corpus=corpus, dimension=dimension: glove_factory(corpus, dimension))

def get_language_model(name):
    """ Get an language model by its name """
    if not __sets.has_key(name):
        raise KeyError('Unknown language model: {}'.format(name))
    return __sets[name]()

def list_language_models():
    """ List all registered language models """
    return __sets.keys()
