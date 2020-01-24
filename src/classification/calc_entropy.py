def calc_entropy(parent: Array, parent_entropy: float, children: list) -> float:
        """Calculates the entropy
    """

    children_entropy = 0
    for child in children:
        children_entropy += (child.shape[0] * child.shape[1]) / (parent.shape[0] * parent.shape[1])
    return parent_entropy - children_entropy
