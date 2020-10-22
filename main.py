import numpy as np
import glob
from qml.representations import generate_fchl_acsf
from qml.utils import NUCLEAR_CHARGE

class XYZReader(object):
    """
    File reader for XYZ format. Adapted from QML: https://github.com/qmlcode/qml
    """

    def __init__(self, filenames=None):
        """
        :param filenames: single filename, list of filenames or a string to be read by glob. e.g. 'dir/*.xyz'
        :type filenames: list or string
        """

        if isinstance(filenames, str):
            if ".xyz" not in filenames:
                filenames = sorted(glob.glob(filenames))
            else:
                filenames = [filenames]

        self._parse_xyz_files(filenames)

    def get_filenames(self):
        """
        Returns a list of filenames in the order they were parsed.
        """
        return self.filenames

    def _parse_xyz_files(self, filenames):
        """
        Parse a list of xyz files.
        """

        n = len(filenames)

        coordinates = []
        elements = []
        files = []

        # Report if there's any error in reading the xyz files.
        try:
            for i, filename in enumerate(filenames):
                with open(filename, "r") as f:
                    lines = f.readlines()

                natoms = int(lines[0])

                if len(lines) % (natoms + 2) != 0:
                    raise SystemExit("1 Error in parsing coordinates in %s" % filename)

                n_snapshots = len(lines) // (natoms + 2)

                if n_snapshots > 1:
                    traj_flag = True
                else:
                    traj_flag = False

                for k in range(n_snapshots):
                    elements.append(np.empty(natoms, dtype='<U3'))
                    coordinates.append(np.empty((natoms, 3), dtype=float))

                    if traj_flag:
                        files.append(filename + "_%d" % k)
                    else:
                        files.append(filename)

                    for j, line in enumerate(lines[k * (natoms+2) + 2: (k+1) * (natoms + 2)]):
                        tokens = line.split()

                        if len(tokens) < 4:
                            raise SystemExit("2 Error in parsing coordinates in %s" % filename)

                        elements[-1][j] = tokens[0]
                        coordinates[-1][j] = np.asarray(tokens[1:4], dtype=float)
        except:
            raise SystemExit("3 Error in reading %s" % filename)

        # Set coordinates and nuclear_charges
        self.coordinates = np.asarray(coordinates)
        self.elements = np.asarray(elements, dtype='<U3')
        self.filenames = files

def create_molecular_representation(coordinates, elements):
    """ Creates permutation, rotation and translation invariant
        atomic representations (10.1063/1.5126701) and converts
        them into a molecular representation.
    """
    nuclear_charges = np.asarray([NUCLEAR_CHARGE[element] for element in elements])
    unique_nuclear_charges = np.unique(nuclear_charges)
    atomic_representations = []
    for snapshot_coordinates in coordinates:
        # Create atomic representations for all atoms in the
        # molecule snapshot
        rep = generate_fchl_acsf(nuclear_charges, snapshot_coordinates,
                    elements=unique_nuclear_charges, nRs2=24, nRs3=20,
                    nFourier=1, eta2=0.32, eta3=2.7, zeta=np.pi, rcut=8.0,
                    acut=8.0, two_body_decay=1.8, three_body_decay=0.57,
                    three_body_weight=13.4, pad=len(elements), gradients=False)

        # Collect for all snapshots
        atomic_representations.append(rep)

    atomic_representations = np.asarray(atomic_representations)

    # Transform the set of atomic representations to molecular ones
    n_snapshots, n_atoms, representation_size = atomic_representations.shape
    n_unique_elements = len(unique_nuclear_charges)
    molecular_representations = \
        np.zeros((n_snapshots, n_unique_elements * representation_size))

    for i, nuclear_charge in enumerate(unique_nuclear_charges):
        # Find indices of the query element
        idx = np.where(nuclear_charges == nuclear_charge)[0]
        # sum all atomic representation of a given element
        atomic_representation_sum = atomic_representations[:,idx].sum(1)
        # Concatenate the vectors together to form the
        # molecular representation
        molecular_representations[:,i*representation_size:(i+1)*representation_size] = \
            atomic_representation_sum

    return molecular_representations


if __name__ == "__main__":
    reader = XYZReader('bifur_traj1.xyz')
    coordinates = reader.coordinates
    elements = reader.elements[0]
    x = create_molecular_representation(coordinates, elements)

    # Then do PCA or whatever
