from collections import defaultdict
import numpy as np
from itertools import combinations
import networkx as nx
from scipy.misc import comb

from pgm.factor import Factor
from pgm.factor import log
from pgm.factor import exp
from pgm.factor import Scalar
from pgm.factor import renormalize
from pgm.factor import get_product_from_list
from pgm.factor import get_product
from pgm.factor import get_marg
from pgm.factor import assign_to_indx


def write_graph(cg, to_name):
    S = set()
    with open('/Users/MattJohnson/Desktop/network.txt', 'w') as f:
        f.write('Source;Target\n')
        for n1 in cg.node:
            for n2 in cg.neighbors(n1):

                n1_factor = ''
                n1_scope = ','.join(cg.node[n1]['scope'])
                if cg.node[n1]['factor'] is not None:
                    n1_factor = ','.join([to_name[s] for s in cg.node[n1]['factor'].scope])
                n1_name = '{0} F({1}) S({2})'.format(n1, n1_factor, n1_scope)

                n2_factor = ''
                n2_scope = ','.join(cg.node[n2]['scope'])
                if cg.node[n2]['factor'] is not None:
                    n2_factor = ','.join([to_name[s] for s in cg.node[n2]['factor'].scope])
                n2_name = '{0} F({1}) S({2})'.format(n2, n2_factor, n2_scope)

                name = ''.join(sorted([n1_name, n2_name]))
                if name in S:
                    continue
                f.write('{0};{1}\n'.format(n1_name, n2_name))
                S.add(name)


def check_rip(cg):
    # Get names of full scope
    scopes = set()
    for name, data in cg.nodes_iter(data=True):
        scopes.update(data['scope'])

    for s in scopes:
        nodes = []
        for name, data in cg.nodes_iter(data=True):
            if s in data['scope']:
                nodes.append(name)
        H = cg.subgraph(nodes)
        print '{0} {1} satisify RIP'.format(s, 'does' if nx.is_connected(H) else 'does not')


def check_calibration(cg, to_index, to_name):
    beliefs = dict()
    for clique_node, data in cg.nodes_iter(data=True):
        indices = data['belief'].scope
        node_names = [to_name[i] for i in indices]

        for i in xrange(len(node_names)):
            factor = data['belief']
            for j in xrange(len(node_names)):
                if i == j:
                    continue
                factor = get_marg(factor, to_index[node_names[j]], use_log_space=True)
            # Renormalize
            val = factor.val / factor.val.sum()
            if node_names[i] not in beliefs:
                beliefs[node_names[i]] = {'val': []}
            beliefs[node_names[i]]['val'].append((clique_node, val))

    for k, v in beliefs.iteritems():
        print k, v['val']


def message_pass(cg, to_name):
    root_node = cg.nodes()[0]
    upward_pass(root_node, cg, to_name)
    downward_pass(root_node, cg, to_name)


def compute_beliefs(cg):
    for n, dict_ in cg.nodes_iter(data=True):
        belief = [dict_['factor']]

        for m in dict_['msg_upward']:
            belief.append(m['msg'])

        for m in dict_['msg_downward']:
            belief.append(m['msg'])

        belief = get_product_from_list(belief, use_log_space=True)
        cg.node[n]['belief'] = belief


def upward_pass(root_node, cg, to_name):
    children = set(cg.neighbors(root_node))
    cg.node[root_node]['children'] = children
    for c in children:
        cg.node[c]['parent'] = root_node
        descend_first_pass(c, cg, to_name)


def downward_pass(root_node, cg, to_name):
    descend_second_pass(root_node, cg, to_name)


def get_msg(from_node, to_node, is_upward, cg, to_name):
    factors = []

    # Use factor from node
    f = cg.node[from_node]['factor']
    if f is not None:
        factors = [cg.node[from_node]['factor']]

    msg_names = []

    if is_upward:
        for d in cg.node[from_node]['msg_upward']:
            factors.append(d['msg'])
            msg_names.append(d['name'])
    else:
        for d in cg.node[from_node]['msg_upward']:
            if d['name'] != '{0}->{1}'.format(to_node, from_node):
                factors.append(d['msg'])
                msg_names.append(d['name'])

        for d in cg.node[from_node]['msg_downward']:
            factors.append(d['msg'])
            msg_names.append(d['name'])

    """
    if len(factors) == 32:
        import pdb; pdb.set_trace()
    """
    msg = get_product_from_list(factors, use_log_space=True)

    s = 'Phi({0})  {1}'.format(from_node, msg_names)
    if is_upward:
        print 'Up {0} -> {1}: {2}'.format(from_node, to_node, s)
    else:
        print 'Down {0} -> {1}: {2}'.format(from_node, to_node, s)

    if not isinstance(msg, Scalar):
        sepset = cg.get_edge_data(from_node, to_node)['sepset']
        scope = msg.scope
        for i in scope:
            name = to_name[i]
            if name not in sepset:
                msg = get_marg(msg, i, use_log_space=True)

    return msg


def ascend(node, parent_node, cg, to_name):
    msg = get_msg(from_node=node,
                  to_node=parent_node,
                  is_upward=True,
                  cg=cg,
                  to_name=to_name)

    msg_name = '{0}->{1}'.format(node, parent_node)
    cg.node[parent_node]['msg_upward'].append({'name': msg_name, 'msg': msg})

    # Is mailbox full?
    if len(cg.node[parent_node]['msg_upward']) == len(cg.node[parent_node]['children']):
        # If false, we've reached the root of the tree
        if cg.node[parent_node]['parent'] is not None:
            ascend(parent_node, cg.node[parent_node]['parent'], cg, to_name)


def descend_first_pass(node, cg, to_name):
    parent_node = cg.node[node]['parent']

    # Get children of node
    children = set(cg.neighbors(node))
    children.remove(parent_node)

    # Reached a leaf node
    if len(children) == 0:
        ascend(node, parent_node, cg, to_name)
    else:
        cg.node[node]['children'] = children
        for c in children:
            cg.node[c]['parent'] = node
            descend_first_pass(c, cg, to_name)


def descend_second_pass(parent_node, cg, to_name):
    # Send message to each child from parent
    for c in cg.node[parent_node]['children']:
        msg = get_msg(from_node=parent_node,
                      to_node=c,
                      is_upward=False,
                      cg=cg,
                      to_name=to_name)

        msg_name = '{0}->{1}'.format(parent_node, c)
        cg.node[c]['msg_downward'].append({'name': msg_name, 'msg': msg})

        descend_second_pass(c, cg, to_name)


def get_clique_graph(elim_order,
                     induced_graph,
                     to_name):

    clique_graph = nx.Graph()
    factors = dict()
    f_count = 1

    # Store factors in dictionary
    for name, dict_ in induced_graph.nodes_iter(data=True):
        factor = dict_['factor']
        factor_scope = set([to_name[i] for i in factor.scope])
        factors[f_count] = {'factor': factor, 'scope': factor_scope}
        f_count += 1

    for node_name in elim_order:
        # Any previous messages that should be included
        # in this scope? (generated from other cliques)
        edges = []
        clique_scope = set()

        for name, dict_ in clique_graph.nodes_iter(data=True):
            # Do not connect to used messages
            if dict_['msg_used']:
                continue
            if node_name in dict_['msg_scope']:
                clique_scope.update(dict_['msg_scope'])
                edges.append(name)
                dict_['msg_used'] = True

        # Find factors with node name
        # in scope
        fs = []
        nodes_used = set()
        factor_names = factors.keys()
        for n in factor_names:
            if node_name in factors[n]['scope']:
                fs.append(factors[n]['factor'])
                nodes_used.add(n)
                del factors[n]

        factor = None
        if len(fs) > 0:
            factor = get_product_from_list(fs, use_log_space=True)
            factor_scope = set([to_name[i] for i in factor.scope])
            clique_scope.update(factor_scope)

        msg_scope = clique_scope.copy()
        msg_scope.remove(node_name)

        attr_dict = {'scope': clique_scope,
                     'msg_scope': msg_scope,
                     'nodes_used': nodes_used,
                     'msg_used': False,
                     'factor': factor,
                     'msg_upward': [],
                     'msg_downward': [],
                     'belief': None,
                     'children': [],
                     'parent': None}
        clique_name = 'C' + str(len(clique_graph) + 1)
        clique_graph.add_node(n=clique_name,
                              attr_dict=attr_dict)

        for e in edges:
            sepset = clique_graph.node[e]['scope'].intersection(clique_scope)
            d = {'sepset': sepset}
            clique_graph.add_edge(e, clique_name, d)

    # See if any cliques are isolated
    # if so, they had non-overlapping scopes
    # So we can place them into any clique
    isolated = []
    C = None
    for name, dict_ in clique_graph.nodes_iter(data=True):
        if len(clique_graph.neighbors(name)) == 0:
            isolated.append(name)
        elif C is None:
            C = name

    for name in isolated:
        factor = clique_graph.node[name]['factor']
        f = clique_graph.node[C]['factor']
        s = clique_graph.node[C]['scope']
        f = get_product(f, factor, use_log_space=True)

        clique_scope = set([to_name[i] for i in f.scope])
        scope = clique_scope.update(s)
        clique_graph.node[C]['factor'] = f
        clique_graph.node[C]['scope'] = scope
        clique_graph.remove_node(name)

    # print("IS_CONNECTED: {0}".format(nx.is_connected(clique_graph)))

    return clique_graph


def get_elim_order(ug):
    induced_graph = ug.copy()
    elim_order = []
    unvisited_nodes = set(induced_graph.nodes(data=False))
    elim_order, induced_graph = _get_elim_order(induced_graph,
                                                unvisited_nodes,
                                                elim_order)
    return elim_order, induced_graph


def _get_elim_order(induced_graph,
                    unvisited_nodes,
                    elim_order):

    if len(unvisited_nodes) == 0:
        return elim_order, induced_graph

    elim_node = None
    global_min_fill_edges = float('inf')
    for node in unvisited_nodes:
        # Get neighboring nodes (exclude ones in elim_order)
        neighbors = set(induced_graph.neighbors(node)).difference(elim_order)

        # Edges among neighbors
        num_edges = 0
        for e1, e2 in combinations(neighbors, 2):
            if induced_graph.has_edge(e1, e2):
                num_edges += 1

        # And if they formed complete subgraph
        # how many edges then?
        max_num_edges = comb(len(neighbors), 2)

        # What is the difference?
        num_fill_edges = max_num_edges - num_edges

        # Is this less than the global minimum so far?
        if num_fill_edges < global_min_fill_edges:
            elim_node = node
            global_min_fill_edges = num_fill_edges

    # TODO: Is there a way to do this
    # within calls to networkx library?
    neighbors = set(induced_graph.neighbors(elim_node)).difference(elim_order)
    for n1, n2 in combinations(neighbors, 2):
        induced_graph.add_edge(n1, n2)

    # Add to elim order
    elim_order.append(elim_node)

    # Node is now visited
    unvisited_nodes.remove(elim_node)

    return _get_elim_order(induced_graph,
                           unvisited_nodes,
                           elim_order)

    return induced_graph, elim_order


def get_moral_graph(dg):
    ug = dg.to_undirected()
    for node in dg.nodes_iter(data=False):
        parents = dg.predecessors(node)
        for r in combinations(parents, 2):
            ug.add_edge(r[0], r[1])
    return ug


class BayesianNetwork(object):
    def __init__(self):
        self._nodes = defaultdict(dict)
        self._index = 0
        self._dg = nx.DiGraph()
        self._to_index = dict()
        self._to_name = dict()

    def load_graph_from_json(self, json_):
        self._dg = nx.DiGraph()
        for node_name, value in json_.iteritems():

            # Have we seen this node name before?
            if node_name not in self._to_index:
                self._to_index[node_name] = self._index
                self._to_name[self._index] = node_name
                self._index += 1

            for p in value['parents']:
                if p not in self._to_index:
                    self._to_index[p] = self._index
                    self._to_name[self._index] = p
                    self._index += 1

            p_indx = [self._to_index[p] for p in value['parents']]
            c_indx = self._to_index[node_name]

            scope = np.array([c_indx] + p_indx)
            card = np.ones(scope.shape[0], dtype=np.int32) * 2

            N = np.prod(card)
            val = np.empty((N,))
            for v in value['values']:
                A = np.array(v['states'])
                A = A.reshape((-1, A.shape[0]))
                i = assign_to_indx(A, card)[0]
                val[i] = v['value']

            # What are the parents?
            f = Factor(scope=scope,
                       card=card,
                       val=val)
            f = log(f)

            var_scope = set([node_name] + value['parents'])
            self.add_node(node_name=node_name,
                          attr_dict={'factor': f,
                                     'scope': var_scope})

            self.add_parents(node_name, value['parents'])

    def __len__(self):
        return len(self._dg)

    def add_parents(self,
                    child,
                    parent_names):
        for p in parent_names:
            self._dg.add_edge(p, child)

    def get_parents(self, node_name):
        return self._dg.predecessors(node_name)

    def get_nodes(self):
        for node_name in self._nodes:
            yield node_name

    def infer(self):
        ug = get_moral_graph(self._dg)
        elim_order, induced_graph = get_elim_order(ug)
        cg = get_clique_graph(elim_order, induced_graph, self._to_name)

        if len(cg) > 1:
            message_pass(cg, self._to_name)
        compute_beliefs(cg)

        # check_rip(cg)
        # write_graph(cg, self._to_name)
        check_calibration(cg, self._to_index, self._to_name)

        # Update random variables for all nodes in network
        for clique_node, data in cg.nodes_iter(data=True):

            indices = data['belief'].scope
            node_names = [self._to_name[i] for i in indices]

            for i in xrange(len(node_names)):
                factor = exp(data['belief'])
                for j in xrange(len(node_names)):
                    if i == j:
                        continue
                    factor = get_marg(factor, self._to_index[node_names[j]], use_log_space=False)

                factor = renormalize(factor)
                self.set_value(node_names[i], 'rv', factor)

    def get_value(self,
                  node_name,
                  key):
        return self._dg.node[node_name][key]

    def set_value(self,
                  node_name,
                  key,
                  value):
        self._dg.node[node_name][key] = value

    def add_node(self,
                 node_name,
                 attr_dict=None):
        self._dg.add_node(n=node_name,
                          attr_dict=attr_dict)
