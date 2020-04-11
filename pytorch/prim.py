#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:36:20 2020

@author: yuzhang
"""

import heapq
import itertools
import math

class GraphNode(object):
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return "Node:{}".format(self.name)
    
    def __repr__(self):
        return "Node:{}".format(self.name)


class Graph(object):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self._construct()
    
    def all_nodes(self):
        return self.nodes
    
    def _construct(self):
        self._adjacent_nodes = {}
        for tail, head, weight in self.edges:
            if tail not in self._adjacent_nodes:
                self._adjacent_nodes[tail] = {}
            self._adjacent_nodes[tail][head] = weight
    
    def adjacent_nodes(self, node):
        if node not in self._adjacent_nodes:
            return list()
        
        return self._adjacent_nodes[node].keys()
    
    def edge_weight(self, u, v):
        if u not in self._adjacent_nodes:
            return math.inf
        
        if v not in self._adjacent_nodes[u]:
            return math.inf
        
        return self._adjacent_nodes[u][v]


class PriorityQueue(object):
    REMOVED = '<removed-task>'
    
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()
        self.node_count = 0

    def add_node(self, graph_node, priority):
        if graph_node in self.entry_finder:
            self.remove_node(graph_node)
        count = next(self.counter)
        entry = [priority, count, graph_node]
        self.entry_finder[graph_node] = entry
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
    
    def valid_node_count(self):
        return self.node_count


class Prim(object):
    def __init__(self, graph):
        self.graph = graph
    
    def minimum_spanning_tree(self, root_node):
        self.dist = {}
        self.parent = {}
        self.tree_nodes = set()
        
        for node in self.graph.all_nodes():
            self.dist[node] = math.inf
            self.parent[node] = None
        self.dist[root_node] = 0
        
        priority_queue = PriorityQueue()
        for node, val in self.dist.items():
            priority_queue.add_node(node, val)
        
        
        while priority_queue.valid_node_count() > 0:
            node = priority_queue.pop_node()
            print("Pop node {}".format(node.name))
            self.tree_nodes.add(node)
            for v in self.graph.adjacent_nodes(node):
                in_tree = v in self.tree_nodes
                edge_weight = self.graph.edge_weight(node, v)
                if (not in_tree) and (edge_weight < self.dist[v]):
                    self.parent[v] = node
                    self.dist[v] = edge_weight
                    priority_queue.add_node(v, edge_weight)
                    
                    
        for node in self.graph.all_nodes():
            parent_node = self.parent[node]
            if parent_node is not None:
                print("edge: {} -> {}: {}".format(
                    parent_node, node, self.graph.edge_weight(parent_node, node)))
        
        return self.dist, self.parent


def construct_undirected_graph(input_edges):
    node_dict = {}
    edges = []
    for tail_node_name, head_node_name, weight in input_edges:
        if tail_node_name not in node_dict:
            node_dict[tail_node_name] = GraphNode(tail_node_name)
        if head_node_name not in node_dict:
            node_dict[head_node_name] = GraphNode(head_node_name)
        
        tail_node = node_dict[tail_node_name]
        head_node = node_dict[head_node_name]
        
        edges.append((tail_node, head_node, weight))
        edges.append((head_node, tail_node, weight))
        
    nodes = list(node_dict.values())
    print(nodes)
    print(edges)
    return Graph(nodes, edges)


graph = construct_undirected_graph(
    [
     ('a', 'b', 4),
     ('a', 'h', 8),
     ('b', 'h', 11),
     ('b', 'c', 8),
     ('c', 'd', 7),
     ('c', 'i', 2),
     ('i', 'h', 7),
     ('i', 'g', 6),
     ('h', 'g', 1),
     ('g', 'f', 2),
     ('c', 'f', 4),
     ('d', 'f', 14),
     ('d', 'e', 9),
     ('e', 'f', 10),
     ]
    )
prim = Prim(graph)
prim.minimum_spanning_tree(graph.all_nodes()[0])