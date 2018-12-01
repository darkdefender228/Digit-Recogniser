#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 18:02:01 2018

@author: romanilechko
"""
import numpy as np

def make_lines(coords, close, far):
    lines = []
    openning = True
    for index in range(len(coords[:-1])):
        if openning:
            if coords[index] + close >= coords[index + 1]:
                lines.append(coords[index])
                openning = False
        else:
            if index == len(coords) - 2 and coords[index] + close >= coords[index + 1]:
                lines.append(coords[index])
                openning = True
            elif coords[index] + far <= coords[index + 1]:
                lines.append(coords[index])
                openning = True
    return np.asarray(lines)

def cropping(y_lines, x_lines, im):
    """
    y - horizontal line for cropping
    x - verical line for cropping
    return - np.ndarray coordinates. bottom y, x, top y, x.
    """
    coords = []
    
    for i, y in enumerate(y_lines[:-1]):
        if i % 2 == 0:
            single_coord = []
            
            for j, x in enumerate(x_lines):
                if j % 2 == 1:
                    single_coord.append(y_lines[i+1])
                    single_coord.append(x + 5)
                    coords.append(np.asarray(single_coord))
                    single_coord = []
                else:
                    single_coord.append(y)
                    single_coord.append(x - 5)
    return np.asarray(coords)