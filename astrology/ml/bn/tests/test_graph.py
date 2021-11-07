import pytest
import json

from pgm.graph import BayesianNetwork
from pgm.graph import get_moral_graph
from pgm.graph import get_elim_order
from pgm.factor import Factor
from pgm.factor import is_equal
import networkx as nx
import numpy as np


@pytest.fixture(scope='module')
def dgraph_gold():
    graph = {1: {'parents': [], 'kv': {'key_1': 'value_1'}},
             2: {'parents': [], 'kv': {'key_2': 'value_2'}},
             3: {'parents': [1, 2], 'kv': {'key_3': 'value_3'}}}
    return graph


@pytest.fixture(scope='module')
def dgraph_test(dgraph_gold):
    # Create graph
    dg = BayesianNetwork()
    for n, v in dgraph_gold.iteritems():
        dg.add_node(node_name=n,
                    attr_dict=v['kv'])
        dg.add_parents(child=n,
                       parent_names=v['parents'])
    return dg


def test_parents(dgraph_test, dgraph_gold):
    for n in dgraph_gold:
        parents = dgraph_test.get_parents(n)
        assert parents == dgraph_gold[n]['parents']


def test_dict_(dgraph_test, dgraph_gold):
    for n in dgraph_test.get_nodes():
        dict_ = dgraph_gold[n]['kv']
        for k, v in dict_.iteritems():
            assert v == dgraph_test.get_value(n, k)


def test_load_from_json():
    dg = BayesianNetwork()
    with open('one_parent_net.json', 'r') as f:
        json_ = json.load(f)
        dg.load_graph_from_json(json_)

    assert len(dg) == 2

    parents = dg.get_parents('parent')
    assert len(parents) == 0

    parents = dg.get_parents('child')
    assert len(parents) == 1

    factor = dg.get_value('parent', 'factor')
    f = Factor(scope=np.array([0]),
               card=np.array([2]),
               val=np.array([0.57, 0.43]))
    assert is_equal(f, factor)

    factor = dg.get_value('child', 'factor')
    f = Factor(scope=np.array([1, 0]),
               card=np.array([2, 2]),
               val=np.array([0.40, 0.60, 0.13, 0.87]))
    assert is_equal(f, factor)


def test_infer_one_parent_child():
    dg = BayesianNetwork()
    with open('one_parent_net.json', 'r') as f:
        json_ = json.load(f)
        dg.load_graph_from_json(json_)

    dg.infer()

    rv = dg.get_value('parent', 'rv')
    assert np.allclose(np.array([0.57, 0.43]), rv.val)

    rv = dg.get_value('child', 'rv')
    assert np.allclose(np.array([0.2839, 0.7161]), rv.val)


def test_2_network():
    dg = BayesianNetwork()
    with open('one_parent_net.json', 'r') as f:
        json_ = json.load(f)
        dg.load_graph_from_json(json_)

    dg.infer()

    rv = dg.get_value('parent', 'rv')
    assert np.allclose(np.array([0.57, 0.43]), rv.val)

    rv = dg.get_value('child', 'rv')
    assert np.allclose(np.array([0.2839, 0.7161]), rv.val)


def test_4_network():
    dg = BayesianNetwork()
    with open('three_parent_net.json', 'r') as f:
        json_ = json.load(f)
        dg.load_graph_from_json(json_)

    with open('three_parent_net_answer.json', 'r') as f:
        answer = json.load(f)

    dg.infer()

    for node in dg._dg.node:
        rv = dg.get_value(node, 'rv')
        assert np.allclose(np.array(answer[node]), rv.val)


def test_16_network():
    dg = BayesianNetwork()
    with open('16_network.json', 'r') as f:
        json_ = json.load(f)
        dg.load_graph_from_json(json_)

    with open('16_network_answer.json', 'r') as f:
        answer = json.load(f)

    dg.infer()

    for node in dg._dg.node:
        rv = dg.get_value(node, 'rv')
        assert np.allclose(np.array(answer[node]), rv.val)


def test_50_network():
    dg = BayesianNetwork()
    with open('50_network.json', 'r') as f:
        json_ = json.load(f)
        dg.load_graph_from_json(json_)

    with open('50_network_answer.json', 'r') as f:
        answer = json.load(f)

    dg.infer()

    for node in dg._dg.node:
        rv = dg.get_value(node, 'rv')
        assert np.allclose(np.array(answer[node]), rv.val), node



def test_get_moral_graph():
    dg = nx.DiGraph()
    edges = [(1, 4), (2, 4),
             (3, 4), (4, 7),
             (5, 7), (6, 7),
             (7, 8)]
    dg.add_edges_from(edges)

    ug = get_moral_graph(dg)
    assert ug.neighbors(1) == [2, 3, 4]
    assert ug.neighbors(2) == [1, 3, 4]
    assert ug.neighbors(3) == [1, 2, 4]
    assert ug.neighbors(4) == [1, 2, 3, 5, 6, 7]
    assert ug.neighbors(5) == [4, 6, 7]
    assert ug.neighbors(6) == [4, 5, 7]
    assert ug.neighbors(7) == [8, 4, 5, 6]
    assert ug.neighbors(8) == [7]


def test_get_elim_order():
    ug = nx.Graph()
    ug.add_edge('1', '2')
    ug.add_edge('2', '3')
    ug.add_edge('3', '4')
    ug.add_edge('3', '5')
    n1 = ug.number_of_edges()
    elim_order, induced_graph = get_elim_order(ug)
    n2 = induced_graph.number_of_edges()
    assert n1 == n2
