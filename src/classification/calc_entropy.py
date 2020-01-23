def calc_entropy(parent: Array, parent_entropy: float, children: list) -> float:
        """Calculates the entropy

    Args:
        parent_size -- a string giving the path to the .txt data file
        parent_entropy -- a 
        children -- 
    Returns:
        data_set -- a data_set object containing both the attributes
            np array and the features np array
    """
    children_entropy = 0
    for child in children:
        children_entropy += (child.shape[0] * child.shape[1]) / (parent.shape[0] * parent.shape[1])
    return parent_entropy - children_entropy