import json
from pgm.elim import get_elim_order
from pgm.graph import BayesianNetwork
from pgm.graph import get_moral_graph


def test_get_elim_order():
    dg = BayesianNetwork()
    with open('three_parent_net.json', 'r') as f:
        json_ = json.load(f)
        dg.load_graph_from_json(json_)
    ug = get_moral_graph(dg._dg)
    I, elim_order = get_elim_order(ug)
    # TODO
    assert True
