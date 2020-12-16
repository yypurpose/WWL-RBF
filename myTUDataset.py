from __future__ import absolute_import
import numpy as np
import os
import random

from dgl.data.dgl_dataset import DGLBuiltinDataset
from dgl.data.utils import loadtxt, save_graphs, load_graphs, save_info, load_info
from dgl import backend as F
from dgl.utils import retry_method_with_fix
from dgl.convert import graph as dgl_graph


class TUDataset(DGLBuiltinDataset):
    _url = r"https://www.chrsmrrs.com/graphkerneldatasets/{}.zip"

    def __init__(self, name, raw_dir='/mnt/data3/heyy/codes/course/data_mining/WWLsquare/dataset', force_reload=False, verbose=False):
        url = self._url.format(name)
        print(force_reload)
        # self.process()
        # self.save()
        # exit()
        super(TUDataset, self).__init__(name=name, url=url,
                                        raw_dir=raw_dir, force_reload=force_reload,
                                        verbose=verbose)
    
    def process(self):
        DS_edge_list = self._idx_from_zero(
            loadtxt(self._file_path("A"), delimiter=",").astype(int))
        DS_indicator = self._idx_from_zero(
            loadtxt(self._file_path("graph_indicator"), delimiter=",").astype(int))
        DS_graph_labels = self._idx_reset(
            loadtxt(self._file_path("graph_labels"), delimiter=",").astype(int))

        g = dgl_graph(([], []))
        g.add_nodes(int(DS_edge_list.max()) + 1)
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])

        node_idx_list = []
        self.max_num_node = 0
        for idx in range(np.max(DS_indicator) + 1):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
            if len(node_idx[0]) > self.max_num_node:
                self.max_num_node = len(node_idx[0])

        self.num_labels = max(DS_graph_labels) + 1
        self.graph_labels = F.tensor(DS_graph_labels)

        self.attr_dict = {
            'node_labels': ('ndata', 'node_labels'),
            'node_attributes': ('ndata', 'node_attr'),
            'edge_labels': ('edata', 'edge_labels'),
            'edge_attributes': ('edata', 'node_labels'),
        }

        for filename, field_name in self.attr_dict.items():
            try:
                data = loadtxt(self._file_path(filename),
                               delimiter=',').astype(float)
                if 'label' in filename:
                    data = F.tensor(self._idx_from_zero(data))
                else:
                    data = F.tensor(data)
                getattr(g, field_name[0])[field_name[1]] = data
            except IOError:
                pass

        self.graph_lists = [g.subgraph(node_idx) for node_idx in node_idx_list]

    def save(self):
        graph_path = os.path.join(self.save_path, 'tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'tu_{}.pkl'.format(self.name))
        label_dict = {'labels': self.graph_labels}
        info_dict = {'max_num_node': self.max_num_node,
                     'num_labels': self.num_labels}
        save_graphs(str(graph_path), self.graph_lists, label_dict)
        save_info(str(info_path), info_dict)

    def load(self):
        graph_path = os.path.join(self.save_path, 'tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'tu_{}.pkl'.format(self.name))
        graphs, label_dict = load_graphs(str(graph_path))
        info_dict = load_info(str(info_path))

        self.graph_lists = graphs
        self.graph_labels = label_dict['labels']
        self.max_num_node = info_dict['max_num_node']
        self.num_labels = info_dict['num_labels']

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'legacy_tu_{}.pkl'.format(self.name))
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True
        return False

    def __getitem__(self, idx):
        """Get the idx-th sample.

        Parameters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor)
            Graph with node feature stored in ``feat`` field and node label in ``node_label`` if available.
            And its label.
        """
        g = self.graph_lists[idx]
        return g, self.graph_labels[idx]

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)


    def _file_path(self, category):
        return os.path.join(self.raw_path, self.name,
                            "{}_{}.txt".format(self.name, category))

    @staticmethod
    def _idx_from_zero(idx_tensor):
        return idx_tensor - np.min(idx_tensor)

    @staticmethod
    def _idx_reset(idx_tensor):
        """Maps n unique labels to {0, ..., n-1} in an ordered fashion."""
        labels = np.unique(idx_tensor)
        relabel_map = {x: i for i, x in enumerate(labels)}
        new_idx_tensor = np.vectorize(relabel_map.get)(idx_tensor)
        return new_idx_tensor

    def statistics(self):
        return self.graph_lists[0].ndata['feat'].shape[1], \
            self.num_labels, \
            self.max_num_node
