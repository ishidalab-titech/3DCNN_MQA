import numpy as np
import sys, tempfile, subprocess
from functools import reduce
from calculate_axis import get_axis
from amino import get_atom_type_array
from Bio import AlignIO
from prody import parsePDB, LOGGER

LOGGER.verbosity = 'none'


def align_fasta(input_pdb_path, target_fasta_path):
    pdb = parsePDB(input_pdb_path)
    input_fasta_path = tempfile.mktemp(suffix='.fasta')
    f = open(input_fasta_path, 'w')
    f.write('>temp\n')
    if len(pdb.select('name CA').getSequence()) < 25:
        return None, None, None
    else:
        f.write(reduce(lambda a, b: a + b, pdb.select('name CA').getSequence()))
        f.close()
        needle_path = tempfile.mktemp(suffix='.needle')
        cmd = ['needle', '-outfile', needle_path, '-asequence', input_fasta_path, '-bsequence', target_fasta_path,
               '-gapopen', '10', '-gapextend', '0.5']
        subprocess.call(cmd)
        needle_result = list(AlignIO.parse(needle_path, 'emboss'))[0]
        input_seq, target_seq = np.array(list(str(needle_result[0].seq))), np.array(list(str(needle_result[1].seq)))
        input_seq, target_seq = input_seq[np.where(target_seq != '-')], target_seq[np.where(input_seq != '-')]
        input_align_indices = np.where(target_seq != '-')[0]
        target_align_indices = np.where(input_seq != '-')[0]
        align_pdb = pdb.select('resindex ' + reduce(lambda a, b: str(a) + ' ' + str(b), input_align_indices))
        return align_pdb, input_align_indices, target_align_indices


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


def make_voxel(input_mol, buffer, width):
    atom_coord = input_mol.getCoords()
    CA_list, C_list, N_list = input_mol.select('name CA').getCoords(), input_mol.select(
        'name C').getCoords(), input_mol.select('name N').getCoords()
    channel = get_atom_type_array(res_name=input_mol.getResnames(), atom_name=input_mol.getNames())
    output = []
    for ca_coord, c_coord, n_coord in zip(CA_list, C_list, N_list):
        axis = get_axis(CA_coord=ca_coord, N_coord=n_coord, C_coord=c_coord)
        atom = atom_coord - ca_coord
        occus = calc_occupancy_bool(atom_coord=atom, channel=channel, axis=axis, buffer=buffer, width=width)
        output.append(occus)
    output = np.array(output, dtype=bool)
    return output


def get_voxel(input_path, target_path, buffer, width):
    input_mol, _, _ = align_fasta(input_pdb_path=input_path, target_fasta_path=target_path)
    if input_mol is not None:
        occus = make_voxel(input_mol=input_mol, buffer=buffer, width=width)
        return occus, input_mol.select('name CA').getResNames(), input_mol.select('name CA').getResnums()
