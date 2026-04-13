import numpy as np
import functools
import math

def _get_min_max(p):
  return p.min(),p.max()

def _get_bounding_box(points: np.ndarray) -> tuple:
    """ 

    ## Arguments:

    - **points (np.ndarray)**:<br />
        _description_

    ## Returns:

    - **tuple**:<br />
        _description_
    """

    l = np.array([_get_min_max(points[:,dim]) for dim in range(3)])
    return l[:,0], l[:,1]

def _pad_grid(
    v1: np.ndarray,
    v2: np.ndarray=None,
    padding: int=5
) -> tuple:
    """

    ## Arguments:

    - **v1 (np.ndarray)**:<br />
        _description_

    - **v2 (np.ndarray, optional):**:<br />
        _description_

    - **padding (int, optional):**:<br />
        _description_

    ## Returns:

    - **tuple**:<br />
        _description_
    """

    v1 = v1 - padding
    if v2 is not None:
        v2 = v2 + padding
        return v1, v2
    else:
        return v1

def _order_coords(
    a: float,
    b: float,
    s: float,
) -> tuple:
    """ 

    ## Arguments:

    - **a (float)**:<br />
        _description_

    - **b (float)**:<br />
        _description_

    - **s (float)**:<br />
        _description_

    ## Returns:

    - **tuple**:<br />
        _description_
    """

    sign = functools.partial(math.copysign, 1)
    if a > b:
        return (abs(b)-s)*sign(b), (abs(a)-s)*sign(a)
    else:
        return (abs(a)-s)*sign(a), (abs(b)-s)*sign(b)

def _get_voxel_centers(
    v1: np.ndarray,
    v2: np.ndarray,
    voxel_size: float
) -> np.ndarray:
    """ 

    ## Arguments:

    - **v1 (np.ndarray)**:<br />
        _description_

    - **v2 (np.ndarray)**:<br />
        _description_

    - **voxel_size (float)**:<br />
        _description_

    ## Returns:

    - **np.ndarray**:<br />
        _description_
    """

    voxel_centers = []
    x1, x2 = _order_coords(v1[0], v2[0], voxel_size/2)
    for x in np.arange(x1, x2, voxel_size):
        y1, y2 = _order_coords(v1[1], v2[1], voxel_size/2)
        for y in np.arange(y1,y2, voxel_size):
            z1, z2 = _order_coords(v1[2], v2[2], voxel_size/2)
            for z in np.arange(z1,z2, voxel_size):
                voxel_centers.append([x,y,z])

    return np.array(voxel_centers)

def _add_unique_density(
    ls: list,
    ds: np.ndarray,
    num: int
):
    """_summary_

    ## Arguments:

    - **ls (list)**:<br />
        _description_

    - **ds (np.ndarray)**:<br />
        _description_

    - **num (int)**:<br />
        _description_

    ## Returns:

    - **_type_**:<br />
        _description_
    """
    unique, counts = np.unique(ls, return_counts=True)

    return unique, (ds*counts)/num



