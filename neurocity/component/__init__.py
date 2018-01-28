#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# vim: expandtab:ts=4:sw=4

# =========================== MRG Copyright Header ===========================
#
# Copyright (c) 2003-2016 University of Oxford. All rights reserved.
# Authors: Mobile Robotics Group, University of Oxford
#          http://mrg.robots.ox.ac.uk
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== MRG Copyright Header ===========================

"""
Created on Fri Dec 16 10:32:11 2016

@author: bewley
"""

import tensorflow as tf
from neurocity.component.layer import Layer
from neurocity.component.model.model import Model

def convert_layer_to_tensor(layer, dtype=None, name=None, as_ref=False):
    if not isinstance(layer, (Layer, Model)):
        return NotImplemented
    return layer.output

tf.register_tensor_conversion_function((Layer, Model),
                                       convert_layer_to_tensor, 199)