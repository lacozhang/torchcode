#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 10:25:13 2020

@author: yuzhang
https://en.wikipedia.org/wiki/Huffman_coding
"""
import heapq
import itertools

class PriorityQueue(object):
    REMOVED = '<removed-task>'
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()
        self.node_count = 0

    def add_node(self, tree_node):
        if tree_node in self.entry_finder:
            self.remove_node(tree_node)
        count = next(self.counter)
        entry = [tree_node.priority(), count, tree_node]
        self.entry_finder[tree_node] = entry
        heapq.heappush(self.pq, entry)
        self.node_count += 1
        return

    def remove_node(self, tree_node):
        entry = self.entry_finder.pop(tree_node)
        entry[-1] = PriorityQueue.REMOVED
        self.node_count -= 1
        return

    def pop_node(self):
        while self.pq:
            priority, count, node = heapq.heappop(self.pq)
            if node is not PriorityQueue.REMOVED:
                del self.entry_finder[node]
                self.node_count -= 1
                return node
        raise KeyError('pop from an empty priority queue')
    
    def valid_node(self):
        return self.node_count


class TreeNode(object):
    def __init__(self, freq, val):
        self.freq = freq
        self.val = val
        self.encoding = ""
        self.parent = self.left = self.right = None
        return
    
    def priority(self):
        return self.freq
    
    def left_child(self):
        return self.left
    
    def right_child(self):
        return self.right
    
    def path_encoding(self):
        return self.encoding

    def set_left_child(self, child):
        self.left = child
    
    def set_right_child(self, child):
        self.right = child
    
    def set_parent(self, node):
        self.parent = node
    
    def append_encoding(self, op):
        self.encoding += op
    
    
class HuffmanCoding(object):
    def __init__(self, sequence):
        self.data = sequence
        self.pri_q = PriorityQueue()
        
    def follow_path(node, op):
        if node is None:
            return
        node.append_encoding(op)
        
        HuffmanCoding.follow_path(node.left_child(), node.path_encoding() + "0")
        HuffmanCoding.follow_path(node.right_child(), node.path_encoding() + "1")
        
    def organize_tree(self):
        for node in self.data:
            self.pri_q.add_node(node)
            
        while self.pri_q.valid_node() > 1:
            print("#node {} before pop".format(self.pri_q.valid_node()))
            node1 = self.pri_q.pop_node()
            node2 = self.pri_q.pop_node()
            print("#node {} after pop".format(self.pri_q.valid_node()))
            
            new_node = TreeNode(node1.priority() + node2.priority(), None)
            new_node.set_left_child(node1)
            new_node.set_right_child(node2)
            node1.set_parent(new_node)
            node2.set_parent(new_node)
            
            self.pri_q.add_node(new_node)
            
            print("#node {} after add".format(self.pri_q.valid_node()))
            print("Iter finished")
    
    def assign_encoding(self):
        assert self.pri_q.valid_node() == 1
        root = self.pri_q.pop_node()
        HuffmanCoding.follow_path(root, "")
    


chars = [
        TreeNode(7, ' '),
        TreeNode(4, 'a'),
        TreeNode(4, 'e'),
        TreeNode(3, 'f'),
        TreeNode(2, 'h'),
        TreeNode(2, 'i'),
        TreeNode(2, 'm'),
        TreeNode(2, 'n'),
        TreeNode(2, 's'),
        TreeNode(2, 't'),
        TreeNode(1, 'l'),
        TreeNode(1, 'o'),
        TreeNode(1, 'p'),
        TreeNode(1, 'r'),
        TreeNode(1, 'u'),
        TreeNode(1, 'x'),
]

hfc = HuffmanCoding(chars)
hfc.organize_tree()
hfc.assign_encoding()

for node in chars:
    print("Node: '{}', encoding: {}".format(node.val, node.path_encoding()))