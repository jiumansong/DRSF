import numpy as np
import random


class Transition:
    def __init__(self, patches_features, label, history_seq, available_mask,
                 action, reward,
                 next_history_seq, next_available_mask):

        self.patches_features = patches_features
        self.label = label
        self.history_seq = history_seq
        self.available_mask = available_mask
        self.action = action
        self.reward = reward
        self.next_history_seq = next_history_seq
        self.next_available_mask = next_available_mask

    def __iter__(self):
        return iter((self.patches_features, self.label, self.history_seq, self.available_mask,
                     self.action, self.reward, self.next_history_seq, self.next_available_mask))


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  
        self.tree = np.zeros(2 * capacity - 1)  
        self.data = np.zeros(capacity, dtype=object)  
        self.data_pointer = 0  

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1 
        self.data[self.data_pointer] = data  
        self.update(tree_idx, priority)  

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:  
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get_leaf(self, value):
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break
            if value <= self.tree[left_child]:
                parent_idx = left_child
            else:
                value -= self.tree[left_child]
                parent_idx = right_child

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  
        self.eps = 1e-5  

    def push(self, patches_features, label, history_seq, available_mask,
             action, reward, next_history_seq, next_available_mask):
        max_priority = max(self.tree.tree[-self.capacity:]) if len(self) > 0 else 1.0 
        transition = Transition(patches_features, label, history_seq, available_mask,
                                action, reward, next_history_seq, next_available_mask)
        self.tree.add(max_priority, transition)

    def sample(self, batch_size, beta=0.4):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority() / batch_size 

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = np.random.uniform(a, b)  
            index, priority, data = self.tree.get_leaf(value)
            batch.append(data)
            indices.append(index)
            priorities.append(priority)

        probabilities = np.array(priorities) / self.tree.total_priority()
        weights = (self.capacity * probabilities) ** -beta  
        weights /= weights.max() 

        return batch, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.eps) ** self.alpha  
            self.tree.update(idx, priority)

    def __len__(self):
        return min(self.capacity, self.tree.data_pointer)
