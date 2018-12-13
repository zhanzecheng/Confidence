# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/3 下午7:53
# @Author  : zhanzecheng
# @File    : dataset.py
# @Software: PyCharm
"""
from src import nn_utils

class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

class Example:

    def __init__(self, confidence, label, decoder_pob, one_hot_type):
        self.confidence = confidence
        self.label = label
        self.src_sent = len(confidence)
        self.decoder_pob = decoder_pob
        self.one_hot_type = one_hot_type

class Batch:
    def __init__(self, examples, cuda=False):
        self.examples = examples
        self.cuda = cuda

        self.confidence = [e.confidence for e in self.examples]
        self.decoder_pob = [e.decoder_pob for e in self.examples]
        self.one_hot_type = [e.one_hot_type for e in self.examples]
        self.src_sents_len = [len(e.confidence) for e in self.examples]
        self.label = [e.label for e in examples]

    def __len__(self):
        return len(self.examples)

    @cached_property
    def src_sents_var(self):
        return nn_utils.to_input_variable(self.confidence,
                                          cuda=self.cuda)

    @cached_property
    def decoder_pob_car(self):
        return nn_utils.to_input_variable(self.decoder_pob,
                                          cuda=self.cuda)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                             cuda=self.cuda)

