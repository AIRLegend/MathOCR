import numpy as np

def beam_search(levels, index=False):
    """ Searchs for the path with the highest cum. probability.

    Parameters:
    - levels: Matrix with shape [BEAM_WIDTH, SEARCH_DEPTH]
    - index: Return the index on the first level belonging to the best path.
    """
    beam_width = len(levels[0])

    sums = np.array([0.0 for i in range(len(levels) ** beam_width)])
    # front pass
    for depth_i, level in enumerate(levels):
        for i, char in enumerate(level):
            c = len(sums) // len(level)
            sums[i * c:(i + 1) * c] *= char[1]

    # cumm probability over all paths calculated. Now, get the
    # total path of the highest one.
    best_path = np.argmax(sums)
    c = best_path
    best_path = [levels[-1][best_path]]  # Reversed best path

    for i in reversed(range(len(levels) - 1)):
        c = c // beam_width
        best_path.append(levels[i][c])

    if not index:
        return list(reversed(best_path))
    else:
        return list(reversed(best_path)), np.argmax(sums)
