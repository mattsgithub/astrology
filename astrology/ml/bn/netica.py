import re
import json


def get_prob(line):
    p = [float(n) for n in re.findall("\d+\.\d+", line)]
    return p


def get_parent_states(line):
    i = line.index('//') + 3
    ps = line[i:]
    ps = [p.strip() for p in ps.split(' ')]
    ps = [p for p in ps if len(p) > 0]
    ps = [p for p in ps if p != ';']
    return ps


def get_cpt(lines):
    try:
        indx = lines[0].index('//', 10)
    except ValueError:
        return None
    parent_order = [p for p in lines[0][indx + 2:].strip().split(' ')]
    parent_order = [p for p in parent_order if len(p) > 0]

    cpt = []
    cpt = dict()
    cpt = {'parent_order': parent_order,
           'row': []}
    # Now, collect cpt
    for l in lines[1:]:
        p = get_prob(l)
        states = get_parent_states(l)
        cpt['row'].append({'states': states,
                           'prob': p})
    return cpt


def get_node(lines):
    indx = lines[0].index('{')
    node_name = lines[0][5:indx].strip()
    parents = []
    prob_lines = []
    parse_prob = False
    cpt = None
    for line in lines:
        if parse_prob:
            if 'whenchanged' in line or "title" in line:
                parse_prob = False
                cpt = get_cpt(prob_lines)
                prob_lines = []
            else:
                prob_lines.append(line)
        elif 'probs' in line:
            parse_prob = True
        elif 'parents' in line:
            i = line.index('(') + 1
            j = line.index(')')
            parents = [p.strip() for p in line[i:j].split(',') if len(p.strip()) > 0]
        elif 'belief' in line:
            i = line.index('(') + 1
            j = line.index(')')
            belief = [float(p) for p in line[i:j].split(',')]

    node = {'name': node_name,
            'parents': parents,
            'belief': belief,
            'cpt': cpt}
    return node


def parse_netica_dne_file(file_path):
    nodes = []
    node_parse = False
    with open(file_path) as f:
        node_lines = []
        for line in f:
            if line.startswith('node'):
                node_parse = True
                node_lines.append(line)
            elif node_parse and line.startswith('\t};'):
                node_parse = False
                node_lines.append(line)
                nodes.append(get_node(node_lines))
                node_lines = []
            elif node_parse:
                node_lines.append(line)
    return nodes


def write_posterior_beliefs(nodes):
    json_ = dict()
    for n in nodes:
        json_[n['name']] = n['belief']
    with open('/Users/MattJohnson/Development/git/pgm/pgm/tests/50_network_answer.json', 'w') as f:
        json.dump(json_, f)


def write_nodes(nodes):
    j = dict()
    for n in nodes:
        dict_ = dict()
        dict_['parents'] = n['parents']
        dict_['values'] = []
        if len(n['parents']) == 0:
            dict_['values'].append({'states': [0], 'value': n['belief'][0]})
            dict_['values'].append({'states': [1], 'value': n['belief'][1]})
        else:
            dict_['parents'] = n['cpt']['parent_order']
            for row in n['cpt']['row']:
                states = []
                for s in row['states']:
                    if s == 'true':
                        states.append(0)
                    elif s == 'false':
                        states.append(1)
                    else:
                        raise ValueError('state {0}'.format(s))
                dict_['values'].append({'states': [0] + states,
                                        'value': row['prob'][0]})
                dict_['values'].append({'states': [1] + states,
                                        'value': row['prob'][1]})
        j[n['name']] = dict_

    with open('/Users/MattJohnson/Development/git/pgm/pgm/tests/50_network.json', 'w') as f:
        json.dump(j, f)

file_path = '/Users/MattJohnson/Development/git/pgm/netica/50_network.dne'
nodes = parse_netica_dne_file(file_path)
write_nodes(nodes)
write_posterior_beliefs(nodes)
