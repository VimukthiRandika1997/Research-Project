import pandas as pd
import numpy as np 

from rdkit import Chem
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data, download_url, extract_gz

import os
import yaml
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder


print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")



class MyMoleculeDataset(Dataset):

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]
    names = {
        'hiv': ['HIV', 'HIV.csv', 'HIV', 0, -1],
        'bace': ['BACE', 'bace.csv', 'bace', 0, 2],
        'bbbp': ['BBBP', 'BBBP.csv', 'BBBP', -1, -2],
    }


    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """

        self.test = test
        self.filename = filename # dataset.csv filename

        # Setting the target name
        if self.filename.split('.')[0].upper() == 'BBBP':
            self.target_name = 'p_np'
            self.read_name = 'smiles'

        elif self.filename.split('.')[0].upper() == 'HIV':
            self.target_name = 'HIV_active'
            self.read_name = 'smiles'

        elif self.filename.split('.')[0].upper() == 'BACE':
            self.target_name = 'Class'
            self.read_name = 'mol'
        else:
            raise ValueError('Not a defined dataset name!!!')

        super(MyMoleculeDataset, self).__init__(root, transform, pre_transform)


    @property
    def raw_dir(self):
        return os.path.join(self.root, self.filename.split('.')[0], 'raw')


    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename


    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]


    @property
    def processed_dir(self):
        """Processed dir for the created graph data objects"""

        return os.path.join(self.root, self.filename.split('.')[0], 'processed')


    def download(self):
        name = self.filename.split('.')[0].lower()
        url = self.url.format(self.names[name][1])
        path = download_url(url, self.raw_dir)
        if self.names[name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)


    def process(self):
        # Read the CSV file
        self.data = pd.read_csv(self.raw_paths[0])
        skipped = 0
        count = 0

        # Loop through each molecule and apply the pre-processing steps
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol[self.read_name])
            
            if mol_obj is None: # Skipping the erroneous mols, which set the mol_obj to None!
                skipped += 1
                continue

            # Get node features
            node_feats = self._get_node_features(mol_obj)
            # Get edge features
            edge_feats = self._get_edge_features(mol_obj)
            # Apply pre-processing 
            edge_feats = self._transform_edge_features(edge_feats)
            # Get adjacency info
            edge_index = self._get_adjacency_info(mol_obj)
            # Get labels info
            label = self._get_labels(mol[self.target_name])

            # Create data object
            data = Data(x=node_feats, 
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        smiles=mol[self.read_name] # wheather smiles or mol based on the dataset
                        ) 
            
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                f'data_test_{count}.pt'))
                count += 1
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                f'data_{count}.pt'))
                count += 1

        metainfo = {
            'no_of_graphs': count,
            'No of processed graphs': count,
            'Skipped molecules': skipped
        }
        with open(os.path.join(self.processed_dir, 'metainfo.yaml'), 'w') as f:
            yaml.dump(metainfo, f)

        print(f'No of processed molecules: {count}')
        print(f'Skipped molecules: {skipped}')


    def _get_node_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        # Loop through each of atoms in the molecule
        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Feature 6: Total Num Hs
            node_feats.append(atom.GetTotalNumHs())
            # Feature 7: Radical Electrons
            node_feats.append(atom.GetNumRadicalElectrons())
            # Feature 8: In Ring
            node_feats.append(atom.IsInRing())
            # Feature 9: Chirality
            node_feats.append(atom.GetChiralTag())

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)

        return torch.tensor(all_node_feats, dtype=torch.float)


    def _get_edge_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]

        Ref: https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Bond.GetBondType
        """
        all_edge_feats = []

        # Loop throught each of the bond in the molecule
        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double) [Single: 1, Double: 2, Triple: 3, Aromatic: 1.5]
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction), if the graph is undirected graph!
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)

        return torch.tensor(all_edge_feats, dtype=torch.float)


    def _get_adjacency_info(self, mol):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return edge_indices


    def _get_labels(self, label):
        """
        Get the graph lable as the target
        """

        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)


    def _transform_edge_features(self, edge_attr_mat):
        """Transform edge features into proper encoding format
            for the training task.

            Especifically, one-hot encoding for bond-type and
            rest of features will be concatenated along the axis=1

            Ref: https://www.scaler.com/topics/data-science/one-hot-encoding/
        """

        # Define the lower and upper limits for the feat: `bond_type`
        # single bond, double bond, triple bond, 3 > aromatic

        # one-hot encoding of type of bonds
        edge_a_norm = edge_attr_mat[:,0].detach().cpu().numpy()
        edge_a_norm = np.expand_dims(edge_a_norm, axis=1)
        edge_a_norm = np.where(edge_a_norm > 3, 1.5, edge_a_norm) # if anything greater than triple bond -> aromatic bond
        
        encoder = OneHotEncoder(categories=[[1.0, 1.5, 2.0, 3.0]], sparse_output=False)
        onehot = encoder.fit_transform(edge_a_norm)

        edge_bond_type = torch.tensor(onehot, dtype=torch.float)
        rest_of_feat = edge_attr_mat[:, 1:]
        new_edge_attr = torch.cat([edge_bond_type, rest_of_feat], dim=1)

        return new_edge_attr


    def _transform_node_features():
        pass


    def len(self):
        with open(os.path.join(self.processed_dir, 'metainfo.yaml'), 'r') as f:
            data_dict = yaml.safe_load(f)

        return data_dict['no_of_graphs']


    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data


