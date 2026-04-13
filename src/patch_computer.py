import numpy as np
import itertools
import jenkspy
from sparse_grid import SparseGrid

def calc_bead_spread(
    tup: tuple,
    grid: SparseGrid
) -> float:
    """_summary_

    ## Arguments:

    - **tup (tuple)**:<br />
    A tuple containing the indices of the voxels and their corresponding densities for a given bead.

    - **grid (SparseGrid)**:<br />
    The sparse grid object containing the grid information.

    ## Returns:

    - **float**:<br />
    The bead spread, which is a measure of how spread out the bead is in the
    grid. It is calculated as the square root of the average squared distance
    of the voxel coordinates from the center of mass of the bead, weighted by
    the densities.
    """

    inds = tup[0]
    den = np.expand_dims(tup[1], axis=-1)
    voxel_coords = np.array([grid.coordinate_to_index(i) for i in inds])
    v_com = np.sum(voxel_coords*den, axis=0)/np.sum(den)
    if np.sum(den) == 0:
        return 0
    bead_spread = np.sqrt(
        np.sum(np.sum(np.square(voxel_coords-v_com)*den, axis=1), axis=0)/
        np.sum(den)
    )

    return bead_spread

def to_array(
    final: list,
    ids: list
) -> np.ndarray:
    """

    ## Arguments:

    - **final (list)**:<br />
        _description_

    - **ids (list)**:<br />
        _description_

    ## Returns:

    - **np.ndarray**:<br />
        _description_
    """

    full_arr = np.zeros((len(ids), len(ids)))
    pairs = itertools.combinations( ids, 2 )
    for i,p in enumerate(pairs):
        a,b = p
        full_arr[ids.index(a),ids.index(b)] = final[i]
        full_arr[ids.index(b),ids.index(a)] = final[i]

    return full_arr

def calc_distance_matrix(
    args: list,
    coords: np.ndarray,
    radius: list
) -> np.ndarray:
    """ 

    ## Arguments:

    - **args (list)**:<br />
        _description_

    - **coords (np.ndarray)**:<br />
        _description_

    - **radius (list)**:<br />
        _description_

    ## Returns:

    - **np.ndarray**:<br />
        _description_
    """

    pairs = itertools.combinations(args, 2)
    mean_dist = []
    for p in pairs:
        surface_distance = (
            np.linalg.norm(coords[:,p[0],:] - coords[:,p[1],:], axis=1)-
            (radius[p[0]] + radius[p[1]])
        )
        mean_dist.append(
            np.mean(surface_distance) if np.mean(surface_distance) >= 0 else 0
        )

    return to_array(mean_dist, args)

def thresh_to_arg(
    bead_spread: np.ndarray,
    low_thresh: float,
    high_thresh: float
) -> list:
    """ 

    ## Arguments:

    - **bead_spread (np.ndarray)**:<br />
        _description_

    - **low_thresh (float)**:<br />
        _description_

    - **high_thresh (float)**:<br />
        _description_

    ## Returns:

    - **list**:<br />
        _description_
    """

    return [
        n for n,i in enumerate(bead_spread)
        if i >= low_thresh and i <= high_thresh
    ]

def get_connected_components(
    arg: list,
    coords: np.ndarray,
    radius: list,
    thresh: float=10,
) -> list:
    """ 

    ## Arguments:

    - **arg (list)**:<br />
        _description_

    - **coords (np.ndarray)**:<br />
        _description_

    - **radius (list)**:<br />
        _description_

    - **thresh (float, optional):**:<br />
        _description_

    ## Returns:

    - **list**:<br />
        _description_
    """

    import networkx as nx
    dist = calc_distance_matrix(arg, coords, radius)
    true_pairs = np.argwhere(dist < thresh)
    l = []
    for tp in true_pairs:
        l.append( (arg[tp[0]], arg[tp[1]]) )
    G = nx.Graph()
    G.add_edges_from(l)
    clusts = []
    for connected_component in nx.connected_components(G):
        clusts.append(list(connected_component))

    return clusts

def get_patches(
    bead_spread: np.ndarray,
    classes: int,
    coords: np.ndarray,
    radius: list
) -> list:
    """ 

    ## Arguments:

    - **bead_spread (np.ndarray)**:<br />
        _description_

    - **classes (int)**:<br />
        _description_

    - **coords (np.ndarray)**:<br />
        _description_

    - **radius (list)**:<br />
        _description_

    ## Returns:

    - **list**:<br />
        _description_
    """

    breaks = jenkspy.jenks_breaks(values=bead_spread, n_classes=(classes*2)+1)
    arg_patches = [
        thresh_to_arg(bead_spread, breaks[i-1], breaks[i])
        for i in range(1,len(breaks))
    ]
    patches = [
        get_connected_components(arg, coords, radius) for arg in arg_patches
    ]

    return patches

def annotate_patches(
    patches: list,
    classes: int,
    ps_names: list,
    num_beads: int
) -> list:
    """

    ## Arguments:

    - **patches (list)**:<br />
        _description_

    - **classes (int)**:<br />
        _description_

    - **ps_names (list)**:<br />
        _description_

    - **num_beads (int)**:<br />
        _description_

    ## Returns:

    - **list**:<br />
        _description_
    """

    annotations = [0]*num_beads
    patch_num = 0
    mapping = np.concatenate(
        (np.arange(1, classes + 1, 1), np.array([1]), np.arange(classes, 0, -1))
    )
    for i,patch in enumerate(patches):
        for j,members in enumerate(patch):
            for member in members:
                tp = 'high' if i < classes else 'low' if i > classes else 'mid'
                lev = mapping[i]
                annotations[member] = [member, ps_names[member], tp, lev, patch_num]
            patch_num += 1

    return annotations

