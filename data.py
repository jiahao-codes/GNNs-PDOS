import os
import csv
import json
import random
import torch
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Dataset, Data
from torch_geometric.data import DataLoader
import networkx as nx
from scipy.ndimage import gaussian_filter1d
from p_tqdm import p_umap  
from tqdm import tqdm  
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*fractional coordinates.*")

random_seed = 999
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)

def smooth(array, sigma = 3):
    return gaussian_filter1d(array, sigma=sigma)
        
class CIFData(Dataset):
    def __init__(self, root_dir, max_num_atoms = 40, train_ratio=0.8, val_ratio=0.1, radius=2.5, dmin=0, step=0.2):
        self.root_dir = root_dir
        self.radius = radius
        self.structures_folder = os.path.join(self.root_dir, 'structure_GGA(+U)')
        self.labels_folder = os.path.join(self.root_dir, 'pdos_GGA(+U)')
        
        data_list = []
        
        self.mp_ids_all = [mp_id.replace('.cif', '') for mp_id in os.listdir(self.structures_folder)]
        print('Screening structures')
        with tqdm(total=len(self.mp_ids_all), unit="sample") as pbar:
            for mp_id in self.mp_ids_all:
                if len(self.load_structure_from_cif(self.structures_folder, mp_id + '.cif')) <= max_num_atoms:
                    data_list.append(mp_id)
                pbar.update(1)  
        self.mp_ids = data_list 
        

        
        random.shuffle(self.mp_ids)  
        
        total_size = len(self.mp_ids)
        indices = list(range(total_size))
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
    
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
    
        self.train_data = [self.mp_ids[i] for i in train_indices]
        self.val_data = [self.mp_ids[i] for i in val_indices]
        self.test_data = [self.mp_ids[i] for i in test_indices]
        


        with open('orbital_electrons.json') as f:
            self.emb = json.load(f)
        self.emb2 = {element: [
                    info.get('1s', 0), 
                    info.get('2s', 0), info.get('2p', 0), 
                    info.get('3s', 0), info.get('3p', 0), info.get('3d', 0), 
                    info.get('4s', 0), info.get('4p', 0), info.get('4d', 0), info.get('4f', 0),
                    info.get('5s', 0), info.get('5p', 0), info.get('5d', 0), info.get('5f', 0),
                    info.get('6s', 0), info.get('6p', 0), info.get('6d', 0), info.get('6f', 0),
                    info.get('7s', 0), info.get('7p', 0)
                ] for element, info in self.emb.items()}

        self.orbital_counts = {}
        for element, info in self.emb.items():

            s_count = info.get('1s', 0) + info.get('2s', 0) + info.get('3s', 0) + info.get('4s', 0) + info.get('5s', 0) + info.get('6s', 0) + info.get('7s', 0)
            p_count = info.get('2p', 0) + info.get('3p', 0) + info.get('4p', 0) + info.get('5p', 0) + info.get('6p', 0) + info.get('7p', 0)
            d_count = info.get('3d', 0) + info.get('4d', 0) + info.get('5d', 0) + info.get('6d', 0)
            f_count = info.get('4f', 0) + info.get('5f', 0) + info.get('6f', 0)
        
            self.orbital_counts[element] = [s_count, p_count, d_count, f_count]


        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def load_structure_from_cif(self, cif_folder, cif_file):
        """
        Load crystal structure from CIF file using pymatgen.
        """
        cif_path = os.path.join(cif_folder, cif_file)
        structure = Structure.from_file(cif_path)
        return structure

    def structure_to_graph(self, structure):
        """
        Convert pymatgen Structure object to a NetworkX graph.
        Nodes represent atoms, and edges represent bonds (bidirectional).
        """
        G = nx.Graph()  # Use undirected graph to ensure bidirectional edges
        
        # Add nodes with atomic information
        for i, site in enumerate(structure):
            G.add_node(i, element=site.species_string, coords=site.coords)
        
        # Add edges based on distance (simple cutoff method)
        cutoff = self.radius  # Use the specified radius as the cutoff
        for i, site1 in enumerate(structure):
            for j, site2 in enumerate(structure):
                if i < j and site1.distance(site2) < cutoff:
                    G.add_edge(i, j, weight=site1.distance(site2))  # Store the distance as the edge weight
        
        return G

    def load_data_single(self, mp_id):

        crystal = self.load_structure_from_cif(self.structures_folder, mp_id + '.cif')

        sga = SpacegroupAnalyzer(crystal)
        
        space_group = sga.get_space_group_symbol()
        space_group_number = sga.get_space_group_number()
        
        # Convert structure to graph (bidirectional edges)
        G = self.structure_to_graph(crystal)
        
        # Initialize edge_index and edge_attr
        edge_index = []
        edge_attr = []
        atom_fea = []
        elec_conf = []
        orbital_counts = []
        
        # Extract node features
        for _, node in G.nodes(data=True):
            element = node['element']
            # print(element)
            elec_conf.append(self.emb2[element])  # Making it a list to be compatible with PyTorch Geometric
            orbital_counts.append(self.orbital_counts[element])
            
            # Use atomic number as the feature
            element_symbol = Element(element)  # For Hydrogen
            atomic_number = element_symbol.number
            atom_fea.append([atomic_number])  # Making it a list to be compatible with PyTorch Geometric
        
        # Extract edges from the graph and create edge_index for bidirectional edges
        for edge in G.edges():
            # Add both directions for undirected edges
            edge_index.append([edge[0], edge[1]])
            edge_index.append([edge[1], edge[0]])  # Add reverse direction as well
            edge_attr.append(G[edge[0]][edge[1]]['weight'])
            edge_attr.append(G[edge[1]][edge[0]]['weight'])  # Same weight for both directions

        #print(edge_attr)    
        #print(type(G[edge[0]][edge[1]]['weight'])) 

        # Convert lists to numpy arrays
        edge_index = np.array(edge_index).T  # Transpose for PyTorch format
        edge_attr = np.array(edge_attr)

        # Expand the distance features using Gaussian expansion
        edge_attr = self.gdf.expand(edge_attr)
        #print(edge_attr.shape)
        pdos_path = os.path.join(self.labels_folder, mp_id + '-pdos.json')
        with open(pdos_path, 'r', encoding='utf-8') as file:
            pdos_dict = json.load(file)

        #print(len(crystal))
        # Convert to numpy arrays if they are lists
        energies = np.array(pdos_dict['energies'])
        pdos_s = smooth(np.array(pdos_dict['s'])/len(crystal))
        pdos_p = smooth(np.array(pdos_dict['p'])/len(crystal))
        pdos_d = smooth(np.array(pdos_dict['d'])/len(crystal))
        pdos_f = smooth(np.array(pdos_dict['f'])/len(crystal))
        pdos = np.stack([pdos_s, pdos_p, pdos_d, pdos_f], axis=0)
        # Correct the band center calculation
        #d_band_center = np.sum(pdos_d * energies) / np.sum(pdos_d)
        p_band_center = np.sum(pdos_p * energies) / np.sum(pdos_p)

        # Convert to tensors for PyTorch
        space_group_number = torch.Tensor([space_group_number])
        space_group_number = space_group_number.unsqueeze(dim=0)
        atom_fea = torch.Tensor(atom_fea)
        elec_conf = torch.Tensor(elec_conf)
        orbital_counts = torch.Tensor(orbital_counts)
        
        edge_attr = torch.Tensor(edge_attr)
        edge_index = torch.LongTensor(edge_index)
        energies = torch.Tensor(energies)
        energies = energies.unsqueeze(dim=0)
        pdos = torch.Tensor(pdos)
        pdos = pdos.unsqueeze(dim=0)
        p_band_center = torch.Tensor([p_band_center])
        p_band_center = p_band_center.unsqueeze(dim=0)        
        #print(f"Atom feature shape: {atom_fea.shape}")
        #print(f"Edge index shape: {edge_index.shape}")
        #print(f"space_group_number shape: {space_group_number.shape}")
        #print(f"space_group_number: {space_group_number}")
        
        # Create the Data object for torch_geometric
        data = Data(mp_id=mp_id, x=atom_fea, edge_index=edge_index, edge_attr=edge_attr, energies = energies, space_group_number = space_group_number, p_band_center = p_band_center, y=pdos, elec_conf = elec_conf, orbital_counts = orbital_counts)

        return data

    def load_data(self, id_list):

        data_list = []
        with tqdm(total=len(id_list), unit="sample") as pbar:
            for mp_id in id_list:
                data = self.load_data_single(mp_id)
                data_list.append(data)
                pbar.update(1)  
        return data_list


def data_loader(dataset, batch_size=128, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    
