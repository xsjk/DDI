import dgl
import dgl.nn
import dgl.udf
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

class RGCN(nn.Module):
    def __init__(self, in_feats_d: int, in_feats_p: int, hidden_size: int, out_feats: int, rel_names: Iterable[str]):
        super().__init__()
        Conv1_dict = {rel_name: dgl.nn.GraphConv(in_feats_d, hidden_size) for rel_name in rel_names}
        Conv1_dict['PDI'] = dgl.nn.GraphConv(in_feats_p, hidden_size)
        Conv1_dict['PPI'] = dgl.nn.GraphConv(in_feats_p, hidden_size)
        self.conv1 = dgl.nn.HeteroGraphConv(Conv1_dict)
        Conv2_dict = {rel_name: dgl.nn.GraphConv(hidden_size, hidden_size) for rel_name in rel_names}
        self.conv2 = dgl.nn.HeteroGraphConv(Conv2_dict, aggregate='sum')
        Conv3_dict = {rel_name: dgl.nn.GraphConv(hidden_size, hidden_size) for rel_name in rel_names}
        self.conv3 = dgl.nn.HeteroGraphConv(Conv3_dict, aggregate='sum')

    def forward(self, graph: dgl.DGLGraph, inputs: dict[str, torch.Tensor]):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        return h

class MLPPredictor(nn.Module):
    def __init__(self, in_features: int, out_classes: int):
        super().__init__()
        self.W1 = nn.Linear(in_features * 2, in_features)
        self.W2 = nn.Linear(in_features, out_classes)

    def apply_edges(self, edges: dgl.udf.EdgeBatch) -> dict[str, torch.Tensor]:
        h_u = edges.src['h']
        h_v = edges.dst['h']
        predict_input = torch.cat([h_u, h_v], 1)
        h = self.W1(predict_input)
        h = F.relu(h)
        score = self.W2(h)
        return {'score': score}

    def forward(self, graph: dgl.DGLGraph, h: torch.Tensor):
        '''
        Parameters
        ----------
        graph : DGLGraph
            The graph that only contains 'Drug' nodes.
        h : Tensor
            Input node features, i.e., the output for each drug by RGCN.
        '''
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']
