import argparse
import glob
import json
import numpy as np
import os
import pandas as pd
from pymatgen.core.structure import Structure
from tqdm import tqdm

def load_or_build_config(data_dir, config_path):
    if os.path.isfile(config_path):
        with open(config_path) as f:
            return json.load(f)
    else:
        return build_config(data_dir, config_path)

def build_config(data_dir, config_path):
    atoms = set()
    for path in tqdm(sorted(glob.glob(os.path.join(data_dir, '*.cif'))), desc="Building Config"):
        crystal = Structure.from_file(path)
        atoms.update(crystal.atomic_numbers)
    unique_z = sorted(atoms)
    config = {
        "atomic_numbers": unique_z,
        "node_vectors": np.eye(len(unique_z)).tolist()  # One-hot encoding
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config

def process(crystal, config, cutoff, max_num_nbr):
    lattice = crystal.lattice.matrix
    volume = crystal.lattice.volume
    coords = crystal.cart_coords
    atoms = crystal.atomic_numbers
    one_hot_vec = np.array(config["node_vectors"])
    z_dict = {z: i for i, z in enumerate(config['atomic_numbers'])}
    
    try:
        atom_fea = np.vstack([one_hot_vec[z_dict[atom]] for atom in atoms])
    except KeyError as e:
        print(f"Atomic number {e} not found in config. Consider rebuilding the config.")
        return None  # Or handle in another appropriate way

    all_nbrs = crystal.get_all_neighbors(cutoff, include_index=True)
    nbr_distances = [np.linalg.norm(nbr.coords - coords[nbr.index]) for nbrs in all_nbrs for nbr in nbrs[:max_num_nbr]]

    features = np.concatenate([lattice.flatten(), atom_fea.flatten(), np.array(nbr_distances).flatten(), [volume]])
    return features

def main(data_dir, output_path, name_database, cutoff, max_num_nbr):
    config_path = f"{output_path}/{name_database.lower()}_config_onehot.json"
    config = load_or_build_config(data_dir, config_path)
    data_files = sorted(glob.glob(os.path.join(data_dir, '*.cif')))
    features_list = []

    for file_path in tqdm(data_files, desc="Processing files"):
        crystal = Structure.from_file(file_path)
        features = process(crystal, config, cutoff, max_num_nbr)
        if features is not None:
            features_list.append(features)

    if features_list:
        max_features_length = max(len(features) for features in features_list)
        feature_labels = [f'feature {i+1}' for i in range(max_features_length)]
        df = pd.DataFrame(features_list, columns=feature_labels)
        df.insert(0, 'cif name', [os.path.basename(f) for f in data_files if features_list])
        df.to_csv(os.path.join(output_path, 'feature_vectors.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CIF files into feature vectors.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing CIF files')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to store output CSV file')
    parser.add_argument('--name_database', type=str, required=True, help='Name of the materials database: MP or OQMD')
    parser.add_argument('--cutoff', type=float, required=True, help='Cutoff distance for neighbors')
    parser.add_argument('--max_num_nbr', type=int, required=True, help='Maximum number of neighbors per atom')
    args = parser.parse_args()
    main(args.data_dir, args.output_path, args.name_database, args.cutoff, args.max_num_nbr)
