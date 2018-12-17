import numpy as np
import sys
import argparse
from htmd.ui import Molecule
from amino import get_atom_type_array, remove_notCNOS
from scipy.sparse import csr_matrix, save_npz


def proj(u, v):
    return u * np.dot(v, u) / np.dot(u, u)


def GS(V):
    V = 1.0 * V  # to float
    U = np.copy(V)
    for i in range(1, V.shape[1]):
        for j in range(i):
            U[:, i] -= proj(U[:, j], V[:, i])
    den = (U ** 2).sum(axis=0) ** 0.5
    E = U / den
    return E


def get_axis(CA_coord, C_coord, N_coord):
    vec_ac = np.array([CA_coord[0] - C_coord[0], CA_coord[1] - C_coord[1], CA_coord[2] - C_coord[2]])
    vec_an = np.array([CA_coord[0] - N_coord[0], CA_coord[1] - N_coord[1], CA_coord[2] - N_coord[2]])
    vec_ab = np.cross(vec_an, vec_ac)
    vec_ac, vec_an, vec_ab = GS(np.array([vec_ac, vec_an, vec_ab]))
    axis = np.array([vec_ac, vec_an, vec_ab])
    return axis


def calc_occupancy_bool(atom_coord, channel, axis, buffer, width):
    length = int(buffer / width)
    occus = np.zeros([length, length, length, channel.shape[1]])
    for i in range(len(atom_coord)):
        h = channel[i]
        box_coord = coord_to_box_coord(atom_coord=atom_coord[i], axis=axis, buffer=buffer, width=width)
        if box_coord is not None:
            occus[box_coord[0]][box_coord[1]][box_coord[2]] = h
    occus = occus.transpose([3, 0, 1, 2])
    return occus


def coord_to_box_coord(atom_coord, axis, buffer, width):
    X = np.dot(np.linalg.inv(axis.T), atom_coord)
    X += np.array([int(buffer / 2), int(buffer / 2), int(buffer / 2)])
    X.astype(int)
    if 0 <= X[0] < buffer and 0 <= X[1] < buffer and 0 <= X[2] < buffer:
        return (X / width).astype(int)
    else:
        return None


def calc_data(pdb_path, buffer, width):
    target_mol = remove_notCNOS(Molecule(pdb_path))
    center_resid = target_mol.get('resid', 'name CA')
    atom_coord = target_mol.get('coords')
    CA_list, C_list, N_list = target_mol.get('coords', 'name CA'), target_mol.get('coords', 'name C'), target_mol.get(
        'coords', 'name N')
    CA_id, C_id, N_id = target_mol.get('resid', 'name CA'), target_mol.get('resid', 'name C'), target_mol.get(
        'resid', 'name N')
    channel = get_atom_type_array(mol=target_mol)
    output = []
    center_resid_list = []
    for i in center_resid:
        index = np.where(CA_id == i)[0][0]
        if (CA_id[index] in C_id) and (CA_id[index] in N_id):
            C_index = np.where(C_id == CA_id[index])[0][0]
            N_index = np.where(N_id == CA_id[index])[0][0]
            CA_coord, C_coord, N_coord = CA_list[index], C_list[C_index], N_list[N_index]
            axis = get_axis(CA_coord=CA_coord, N_coord=N_coord, C_coord=C_coord)
            occus = calc_occupancy_bool(atom_coord=atom_coord - CA_coord, channel=channel, axis=axis, buffer=buffer,
                                        width=width)
            output.append(occus)
            center_resid_list.append(i)

    output = np.array(output, dtype=bool)
    return output, target_mol.get('resname', 'name CA'), target_mol.get('resid', 'name CA')


def save_occus(pdb_path, output_path, buffer, width):
    occus, _, _ = calc_data(pdb_path=pdb_path, buffer=buffer, width=width)
    np.save(output_path, occus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict ')
    parser.add_argument('--input_path', '-i', help='Input data path')
    parser.add_argument('--output_path', '-o', help='Output path')
    args = parser.parse_args()
    save_occus(pdb_path=args.input_path, output_path=args.output_path, buffer=28, width=1)
