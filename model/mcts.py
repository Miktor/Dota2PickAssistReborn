import numpy as np
from typing import List, Tuple

# Monte Carlo Tree Search
# Dirichlet noise parameters for adding noise to select probabilities
MCTS_DIRICHLET_EPSILON = 0.2
MCTS_DIRICHLET_ALPHA = 0.03

# This regulates exploration, 0 - no exploration, 1 - maximum exploration
MCTS_C_PUCT = 1.0


class MCTSNode(object):
    def __init__(self, state):
        self.state = state  # Node payload
        self.children = None  # Array of child nodes
        self.actions = None  # Possible actions from this node
        self.is_expanded = False  # Is node expanded
        self.value = None  # Value of the expanded node

        # Placeholders for numpy arrays
        # Each array represents some values for each possible action (edge)
        self.edge_q = None  # Average edge value
        self.edge_w = None  # Total edge value
        self.edge_p = None  # Edge probability
        self.edge_n = None  # Number of edge visitations

    def select(self, game_model, path_nodes: List['MCTSNode'],
               path_edge_indices: List[int]) -> Tuple['MCTSNode', List['MCTSNode'], List[int]]:
        # If current node is not expanded or current node is terminal (no actions)
        # then select finishes - we found the leaf node
        # Ends the recursion
        if not self.is_expanded or not self.actions:
            return self, path_nodes, path_edge_indices

        # Otherwise, walk the tree by selecting actions with max Q + U
        # Find the child node and action with max Q + U
        n_sqrt_sum = np.sqrt(np.sum(self.edge_n))
        n_sqrt_sum = np.maximum(n_sqrt_sum, 1.0)  # Avoid U == 0, if all N(s,b) == 0

        # Add Dirichlet noise to move probabilities
        p = (1.0 - MCTS_DIRICHLET_EPSILON) * self.edge_p + \
            MCTS_DIRICHLET_EPSILON * np.random.dirichlet([MCTS_DIRICHLET_ALPHA] * len(self.edge_p))

        u = MCTS_C_PUCT * p * n_sqrt_sum / (1.0 + self.edge_n)
        selected = np.argmax(self.edge_q + u)  # type: int

        # If we don't have the state for the successor node yet
        # Then get the state by taking action from this state
        # This "lazy loading" of the child node for argmax(Q + U)
        # and prevents evaluating each successor
        if self.children[selected] is None:
            child_state = game_model.get_state_for_action(self.state, self.actions[selected])
            child_node = MCTSNode(child_state)
            self.children[selected] = child_node

        # Continue search from the child node
        # by calling select of the child node
        # Return the full path (both nodes and edges) of select
        return self.children[selected].select(
            game_model, path_nodes=path_nodes + [
                self,
            ], path_edge_indices=path_edge_indices + [
                selected,
            ])

    def expand(self, game_model, estimator):
        # Get actions that we can take from the current state
        # Note: next states will be evaluated on-demand in select
        # to prevent evaluating nodes with low probabilities
        # (which will no likely to be used)
        self.actions = game_model.get_actions_for_state(self.state)

        # State is terminal (no further actions), node can't have any edges so
        # just predict node's value and return
        if not self.actions:
            _, self.value = estimator.predict(self.state, self.actions)
            return

        # Predict with any kind of estimator the probabilities
        # over action given current state
        self.edge_p, self.value = estimator.predict(self.state, self.actions)

        # Add child edges and node-placeholders
        action_num = len(self.actions)
        self.edge_q = np.zeros(action_num, dtype=np.float32)
        self.edge_w = np.zeros(action_num, dtype=np.float32)
        self.edge_n = np.zeros(action_num, dtype=np.uint8)
        self.children = np.full(action_num, None, dtype=np.object)

        # Mark this node as expanded
        self.is_expanded = True

    def update_edge(self, edge_idx: int, value: float):
        # Updates the value of the edge
        # Incrementing number of visits (n)
        self.edge_n[edge_idx] += 1

        # Adding value to the w (total value of the children nodes)
        self.edge_w[edge_idx] += value

        # Updating q to average value of the children nodes
        self.edge_q[edge_idx] = self.edge_w[edge_idx] / self.edge_n[edge_idx]

    def choose_action(self, tau=1.0, deterministic=False) -> Tuple[list, 'MCTSNode', np.ndarray]:
        # Select an action with some policy from the current state
        # And return chosen action and next state

        # Deterministic policy: select edge with max visits
        if deterministic:
            idx = np.argmax(self.edge_n)  # type: int
            probabilities = np.zeros(len(self.actions))
            probabilities[np.argmax(self.edge_n)] = 1.0
            return self.actions[idx], self.children[idx], probabilities

        # Probabilistic policy: select edge with weights of N^(1/tau)
        exp_n = np.power(self.edge_n, 1.0 / tau)
        probabilities = exp_n / exp_n.sum()
        idx = np.random.choice(len(self.actions), p=probabilities)
        return self.actions[idx], self.children[idx], probabilities

    def run(self, game_model, estimator, simulations: int):
        # Run select, expand and propagate multiple times
        for sim_i in range(simulations):
            # Select target node to expand or terminal node, and get the path edges to target node from this node
            node, path_nodes, path_edges = self.select(game_model, [], [])

            # If node is not expanded
            if not node.is_expanded:
                # Expand the node - it will create edges and get value for the node
                node.expand(game_model, estimator)

            # Update N (visits), W (total value), and Q (average value)
            # for the whole path from this node to target, even if target node is terminal
            for node, edge_idx in zip(path_nodes, path_edges):
                node.update_edge(edge_idx, node.value)


if __name__ == '__main__':
    pass
