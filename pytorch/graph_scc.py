#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:16:19 2020

@author: yu zhang
"""

class Node(object):
    def __init__(self, node_name, node_scc_idx):
        self.scc_idx = node_scc_idx
        self.index = None
        self.low_link = None
        pass
    
    def set_scc_idx(self, scc_idx):
        pass
    
    

class Graph(object):
    def __init(self, graph):
        pass
    
    def add_node(self, node):
        pass
    
    def reverse_graph(self):
        """
        create a new graph by reverse all edge directions in current graph.
        """
        pass

    def neighbours(self, node):
        """
        return all the nodes can be reached from `node`
        """
        pass
    
    def nodes(self):
        """
        return all the nodes in this graph
        """
        pass
    
    def assign_scc_idx(self, node, scc_idx):
        """
        Assign scc index to node in graph.
        """
        pass
    
    def add_edge(self, tail, head):
        """
        Add edge to graph tail -> head.
        """
        pass
        
    
    
    
class Kosaraju(object):
    def __init__(self, graph):
        self.graph = graph
        
    def graph_dfs(self, graph, node):
        if node in self.visited:
            return
        else:
            self.visited.add(node)
            for v in graph.neighbours(node):
                self.graph_dfs(graph, v)
            
            # pre-append current node to list.
            self.node_visited.insert(0, node)
    
    def assign_scc(self, graph, node, root):
        if node in self.node_assigned:
            return
        else:
            self.node_assigned.add(node)
            graph.assign_scc_idx(node, root)
            for v in graph.neighbours(node):
                self.assign_scc(graph, v, root)
    
    def find_scc(self):
        self.visited = set([])
        self.node_visited = []
        
        for node in self.graph.nodes():
            self.graph_dfs(self.graph, node)
        
        self.node_assigned = set([])
        reverse_graph = self.graph.reverse_graph()
        for node in self.node_visited:
            self.assign_scc(reverse_graph, node, node.scc_idx)
    
    
class Tarjan(object):
    def __init__(self, graph):
        self.graph = graph
        
    def assign_scc(self, node):
        node.index = self.index
        node.low_link = self.index        
        self.index += 1
        self.stack.append(node)
        node.on_stack = True
        
        for v in self.graph.neighbours(node):
            if v.index is None:
                self.assign_scc(v)
                node.low_link = min(v.low_link, node.low_link)
            elif v.on_stack:
                node.low_link = min(node.low_link, v.index)
        
        
        if node.low_link == node.index:
            while True:
                w = self.stack.pop()
                w.on_stack = False
                self.graph.assign_scc_idx(w, node.scc_idx)
                
                if w is node:
                    break
        
        
    def find_scc(self):
        self.index = 0
        self.stack = []
        
        for node in self.graph.nodes():
            self.assign_scc(node)