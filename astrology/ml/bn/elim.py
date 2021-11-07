def get_min_fill_cost(R, n):
    """If n were eliminated,
    how many fill-ins would this introduce?
    """
    cost = 0

    # List of neighbors
    N = R.neighbors(n)
    for i in xrange(len(N)):
        for j in xrange(i + 1, len(N)):
            if N[j] not in R[N[i]]:
                cost += 1
    return cost


def get_elim_order(G):
    """Greedy search algorithm
    Given an undirected graph, find
    a elim order and return
    clique graph

    Uses min fill as cost function
    """
    I = G.copy()
    R = G.copy()
    elim_order = []

    while len(R) > 0:
        min_cost = float('inf')
        opt_node = None

        for n in R:
            cost = get_min_fill_cost(R, n)
            if cost < min_cost:
                min_cost = cost
                opt_node = n

        elim_order.append(opt_node)

        for n1 in R[opt_node]:
            for n2 in R[opt_node]:
                if n1 != n2:
                    I.add_edge(n1, n2)
                    R.add_edge(n1, n2)

        # Remove opt_node from induced graph
        R.remove_node(opt_node)
        for n in R:
            if opt_node in R[n]:
                R.remove_node(opt_node)

    return I, elim_order
