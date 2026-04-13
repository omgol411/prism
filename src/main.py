import os
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool
from sparse_grid import SparseGrid
from bead_density import BeadDensity
from patch_computer import calc_bead_spread, get_patches, annotate_patches
from pdb_parser import parse_all_struct
from utils import _get_bounding_box
import argparse
import tqdm

def main_density_calc(
    i: int,
    coords: np.ndarray,
    mass: np.ndarray,
    radius: np.ndarray,
    grid: SparseGrid,
    voxel_size: float,
    n_breaks: int
) -> np.ndarray:
    """ Calculate the density for the i-th bead from its coordinates.

    ## Arguments:

    - **i (int)**:<br />
        Bead index for which the density is being calculated.

    - **coords (np.ndarray)**:<br />
        Coordinates of the beads across all models (frames).

    - **mass (np.ndarray)**:<br />
        Mass of the beads.

    - **radius (np.ndarray)**:<br />
        Radius of the beads.

    - **grid (SparseGrid)**:<br />
        SparseGrid object containing the grid information.

    - **voxel_size (float)**:<br />
        The voxel size used for density calculation.

    - **n_breaks (int)**:<br />
        The number of breaks to use for cDist calculation.

    ## Returns:

    - **tuple**:<br />
    """

    bead_density = BeadDensity(
        coords.shape[0],
        grid=grid,
        voxel_size=voxel_size,
        padding=4,
        kernel_type='Spherical'
    )
    # Obtain min-max coords for each bead across all models to construct a kernel.
    # k1 --> min xyz coords of kernel; k2 --> max xyz coords of kernel.
    k1, k2 = _get_bounding_box(points=coords[:,i,:])
    bead_density.construct_kernel(k1=k1, k2=k2)

    return bead_density.return_density_opt(
        points=coords[:,i,:],
        radius=radius[i],
        mass=mass[i],
        n_breaks=n_breaks
    )

def min_max_scale(v: np.ndarray) -> np.ndarray:
    """ Perform min-max scaling on the input array v to scale the values between
    0 and 1.

    ## Arguments:

    - **v (np.ndarray)**:<br />
        The input array to be scaled.

    ## Returns:

    - **np.ndarray**:<br />
        The min-max scaled array.
    """

    return (v - np.min(v)) / (np.max(v) - np.min(v))

def get_file_type(input: str) -> str:
    """ Determine the type of the input file (e.g., 'pdb', 'rmf3') by checking
    the extensions of files in the input folder.

    ## Arguments:

    - **input (str)**:<br />
        The input file or folder path.

    ## Returns:

    - **str**:<br />
        The type of the file (e.g., 'pdb', 'rmf3').
    """

    folder_walk = os.walk(input)
    i = 2
    ret=False
    while ret == False:
        file_in_folder = next(folder_walk)[i][0]
        if os.path.splitext(file_in_folder)[-1] == '.pdb':
            file_type = 'pdb'
            ret=True
        elif os.path.splitext(file_in_folder)[-1] == '.rmf3':
            file_type = 'rmf3'
            ret=True
        i = i+1

    return file_type

def get_bead_spread(
    i: int,
    coords: np.ndarray,
    mass: np.ndarray,
    radius: np.ndarray,
    grid: SparseGrid,
    voxel_size: float,
    n_breaks: int
) -> float:
    """ Worker function to calculate the spread for the i-th bead.

    ## Arguments:

    - **i (int)**:<br />
        Bead index for which the spread is being calculated.

    - **coords (np.ndarray)**:<br />
        Coordinates of the beads across all models (frames).

    - **mass (np.ndarray)**:<br />
        Mass of the beads.

    - **radius (np.ndarray)**:<br />
        Radius of the beads.

    - **grid (SparseGrid)**:<br />
        SparseGrid object containing the grid information.

    - **voxel_size (float)**:<br />
        The voxel size used for density calculation.

    - **n_breaks (int)**:<br />
        The number of breaks to use for cDist calculation.

    ## Returns:

    - **float**:<br />
        The spread value for the i-th bead.
    """

    density = main_density_calc(
        i=i,
        coords=coords,
        mass=mass,
        radius=radius,
        grid=grid,
        voxel_size=voxel_size,
        n_breaks=n_breaks,
    )
    spread = calc_bead_spread(tup=density, grid=grid)

    return spread

def write_bead_precision_info(
    ps_names: list,
    output_dir: str,
    prec_info: list,
    txt_fname: str
) -> None:
    """ Write the bead precision information to a text file.
    The text file will contain the level of precision, bead indices, and bead
    names for each bead in the low or high precision category.

    ## Arguments:

    - **ps_names (list)**:<br />
        List of bead names.

    - **output_dir (str)**:<br />
        The directory where the output files will be saved.

    - **prec_info (list)**:<br />
        The precision information for each bead.

    - **txt_fname (str)**:<br />
        The name of the output text file.
    """

    with open(os.path.join(output_dir, f"{txt_fname}.txt"), "w") as fl:
        lev = 1
        fl.write("Level" + "\t" + "Bead Indices" + "\t" + "Bead Names")
        fl.write("\n")
        for level in prec_info:
            for l in level:
                fl.write(str(lev))
                fl.write("\t")
                fl.write(",".join(str(item) for item in l))
                fl.write("\t")
                fl.write(",".join(ps_names[name] for name in l))
                fl.write("\n")
            lev=lev+1

def run_prism(
    coords: np.ndarray,
    mass: np.ndarray,
    radius: np.ndarray,
    ps_names: list,
    args: argparse.Namespace,
    output_dir: str | None = None,
) -> None:
    """ Main function to run the PrISM pipeline. It calculates the bead spread
    for each bead, obtains patches based on the spread values, and annotates
    the patches for low, medium, and high precision.

    ## Arguments:

    - **coords (np.ndarray)**:<br />
        XYZ coordinates of beads across all models (frames).

    - **mass (np.ndarray)**:<br />
        Mass of the beads.

    - **radius (np.ndarray)**:<br />
        Radius of the beads.

    - **ps_names (list)**:<br />
        List of bead names.

    - **args (argparse.Namespace)**:<br />
        Command-line arguments passed to the function.

    - **output_dir (str | None, optional):**:<br />
        The directory where the output files will be saved.
        If None, it will use the output directory specified in args.
        Default is None.
    """

    if output_dir is None:
        output_dir = args.output

    models = round(args.models*coords.shape[0])
    if args.models != 1:
        selected_models = np.random.choice(
            coords.shape[0], models, replace=False
        )
        coords = coords[selected_models]

    print("Number of Models = {}".format(coords.shape[0]))
    print("Number of Beads = {}".format(coords.shape[1]))

    # Create the grid.
    grid = SparseGrid(voxel_size=args.voxel_size)
    grid.create_grid(coords)
    # Not padding the grid.
    grid.pad_grid(0)

    bead_spread = []
    cores_ = min(os.cpu_count() - 1, args.cores)
    chunksize, extra = divmod(coords.shape[1], cores_ * 4)
    if extra:
        chunksize += 1

    _part_func = partial(
        get_bead_spread,
        coords=coords,
        mass=mass,
        radius=radius,
        grid=grid,
        voxel_size=args.voxel_size,
        n_breaks=args.n_breaks
    )

    with Pool(args.cores) as p:
        for spread in tqdm.tqdm(p.imap(
            _part_func,
            range(0, coords.shape[1] ),
            chunksize=chunksize
        )):
            bead_spread.append( spread )

    bead_spread = min_max_scale(np.array(bead_spread))
    print('Bead Spread calculation done')
    os.makedirs(output_dir, exist_ok=True)

    # Save the bead_spread values.
    if args.return_spread == 1:
        bead_spread_txt = os.path.join(
            output_dir,
            f"bead_spreads_cl{args.classes}.txt"
        )
        with open(bead_spread_txt, "w") as fl:
            for bs in bead_spread:
                fl.write('{:0.3f}'.format(bs))
                fl.write("\n")

    # Obtain patches for all the beads.
    patches = get_patches(
        bead_spread=bead_spread,
        classes=args.classes,
        coords=coords,
        radius=radius,
    )
    # Annotate the patches for low-med-high precision.
    annotated_patches = annotate_patches(
        patches=patches,
        classes=args.classes,
        ps_names=ps_names,
        num_beads=coords.shape[1],
    )
    high_prec, low_prec = patches[:args.classes], patches[args.classes+1:]

    annot_df = pd.DataFrame(
        np.array(annotated_patches),
        columns = ['Bead', 'Bead Name', 'Type', 'Class', 'Patch']
    )
    annot_df['Patch'] = pd.to_numeric(annot_df["Patch"])
    annot_df.sort_values(['Patch'], ascending=[True])
    annotations_txt = os.path.join(
        output_dir,
        f"annotations_cl{args.classes}.txt"
    )
    annot_df.to_csv(annotations_txt, index=None)

    write_bead_precision_info(
        ps_names=ps_names,
        output_dir=output_dir,
        prec_info=low_prec,
        txt_fname=f"low_prec_cl{args.classes}"
    )
    write_bead_precision_info(
        ps_names=ps_names,
        output_dir=output_dir,
        prec_info=high_prec,
        txt_fname=f"high_prec_cl{args.classes}"
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser("PrISM")
    parser.add_argument(
        "--input",
        "-i",
        help="npz file or folder containing necessary files",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--input_type",
        "-t",
        help="Type of input: npz/pdb/cif/rmf/dcd.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--voxel_size",
        "-v",
        help="Voxel size for density calculations",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--return_spread",
        "-rs",
        help="Return the spread bead_spread",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--classes",
        "-cl",
        help="Number of classes(1,2,3)",
        default=2,
        choices=[1,2,3],
        type=int,
    )
    parser.add_argument(
        "--cores",
        "-co",
        help="Number of cores to use",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--models",
        "-m",
        help="Percentage of total models to use",
        default=1,
        type=float,
    )
    parser.add_argument(
        "--n_breaks",
        "-n",
        help="Number of breaks to use for cDist calculation",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--resolution",
        help="The resolution at which to sample the beads for rmf input",
        default=30,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--subunit",
        help="Subunit that needs to be sampled for rmf input",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--selection",
        help='File containing dictionary of selected subunits and residues for rmf input',
        default=None,
        required=False,
    )
    args = parser.parse_args()

    if args.input_type == 'npz':
        arr = np.load(args.input)
        coords = arr['arr_0']
        mass = arr['arr_1']
        radius = arr['arr_2']
        ps_names = arr['arr_3']
        del arr
        run_prism(
            coords=coords,
            mass=mass,
            radius=radius,
            ps_names=ps_names,
            args=args,
        )

    elif args.input_type == "pdb":
        coords, mass, radius, ps_names = parse_all_struct(
            folder=args.input,
            _type="pdb",
        )
        run_prism(
            coords=coords,
            mass=mass,
            radius=radius,
            ps_names=ps_names,
            args=args,
        )

    elif args.input_type == "cif":
        coords, mass, radius, ps_names = parse_all_struct(
            folder=args.input,
            _type="cif",
        )
        run_prism(
            coords=coords,
            mass=mass,
            radius=radius,
            ps_names=ps_names,
            args=args,
        )

    elif args.input_type == "ihm":
        from ihm_parser import parse_ihm_models
        parse_ihm_models(args=args)

    elif args.input_type == "rmf":
        from rmf_parser import parse_all_rmfs
        coords, mass, radius, ps_names = parse_all_rmfs(
            path=args.input,
            resolution=args.resolution,
            subunit=args.subunit,
            selection=args.selection
        )
        run_prism(
            coords=coords,
            mass=mass,
            radius=radius,
            ps_names=ps_names,
            args=args,
        )

    elif args.input_type == "dcd":
        from dcd_parser import parse_all_dcds
        coords, mass, radius, ps_names = parse_all_dcds(
            folder=args.input,
            resolution=args.resolution,
            subunit=args.subunit,
            selection=args.selection
        )
        run_prism(
            coords=coords,
            mass=mass,
            radius=radius,
            ps_names=ps_names,
            args=args,
        )