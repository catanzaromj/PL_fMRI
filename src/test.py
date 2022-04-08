#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:26:43 2022

@author: mike
"""
from src.landscapes import construct_landscapes, pad_flatten_landscape_values

# from src.permutation_test import permutation_test


pl_0508 = construct_landscapes("0508", 0)

# p = permutation_test(pl_0508, ["random", "beat"])

# pl_0508_padded = pad_flatten_landscape_values(pl_0508)
