#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:27:12 2020

@author: yuzhang
https://en.wikipedia.org/wiki/Activity_selection_problem
"""

class Activity(object):
    def __init__(self, start, finish, weight):
        self.start = start
        self.finish = finish
        self.weight = weight
        return
    
def binary_search(array, val):
    "find left most activity which finish time larger/equal than val"
    s = 0
    e = len(array)
    while s < e:
        mid = (s+e)/2
        if val >= array[mid].finish:
            s = mid + 1
        else:
            e = mid
    return s
    

    
def weighted_activity_selection(activities):
    activities.sort(key=lambda x: x.finish)
    optimal_solution = {}
    optimal_solution[-1] = 0
    for idx, act in enumerate(activities):
        i = binary_search(activities, act.start)
        if activities[i].finish > act.start:
            i -= 1
        optimal_solution[idx] = max(act.weight + optimal_solution[i], optimal_solution[idx-1])
    
    return optimal_solution[-1]



