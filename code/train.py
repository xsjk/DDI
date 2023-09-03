import numpy as np
import pandas as pd
import pickle as pkl

from argparse import ArgumentParser

import dgl
import dgl.nn
import dgl.data

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from model import *
from utils import *

DEFAULT_HIDDEN_SIZE = 512

class DDIDataLoader:

    def __init__(self, graph: dgl.DGLGraph, train_eid: np.ndarray, batch_size: int = 1024) -> None:
        '''
        Parameters
        ----------
        graph : DGLGraph
            The heterogeneous graph that contains 'Drug', 'Protein' nodes and 'DDI', 'DPI', 'PDI', 'PPI' edges.
            The graph should not be bidirected.
        train_eid : np.ndarray
            The edge indices of the DDI graph used for training.
        batch_size : int
            The batch size of the dataloader.
        '''
        self.graph = graph
        self.dataloader = dgl.dataloading.DataLoader(
            self.graph['Drug', : , 'Drug'], 
            indices=train_eid,
            graph_sampler=dgl.dataloading.as_edge_prediction_sampler(
                sampler=dgl.dataloading.NeighborSampler([-1]),
                negative_sampler=dgl.dataloading.negative_sampler.GlobalUniform(1),
                # exclude='self',
            ),
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=False,
            num_workers=1,
        )

    def __iter__(self):
        with self.dataloader.enable_cpu_affinity():
            for input_nodes, pos_pair_graph, neg_pair_graph, blocks in self.dataloader:
                pos_pair_graph = to_bidirected(pos_pair_graph)
                neg_pair_graph = to_bidirected(neg_pair_graph)
                ddi_graph_split = split_etype(blocks[0])
                graph = combine_graphs(ddi_graph_split, self.graph['DPI'], self.graph['PDI'], self.graph['PPI'])
                graph.ndata['feature'] = self.graph.ndata['feature']
                graph = to_bidirected(graph)
                yield input_nodes, pos_pair_graph, neg_pair_graph, graph


class Model(pl.LightningModule):

    def __init__(self, 
                 pkl_path: str = '../dataset.pkl', 
                 hidden_size=DEFAULT_HIDDEN_SIZE, 
                 learning_rate: float = 1e-3,
                 minimum_sample_count: int = 10,
                 batch_size: int = 1024
                 ):
        super().__init__()

        # save hyperparameters
        self.pkl_path = pkl_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_hyperparameters()

        # load data
        self.data = pkl.load(open('../dataset.pkl', 'rb'))
        self.data['Drugs'] = self.data['Drugs'].reset_index().reset_index().set_index('Drug_ID')
        self.data['Proteins'] = self.data['Proteins'].reset_index().reset_index().set_index('Protein_ID')
        self.graph = get_graph(self.data)
        self.ndata = self.graph.ndata
        self.num_nodes_dict = {ntype: self.graph.number_of_nodes(ntype) for ntype in self.graph.ntypes}
        self.ddi_graph = self.graph['Drug', :, 'Drug']


        # drop the type which has not enough samples
        minimum_sample_count = 10
        types = pd.Series(self.ddi_graph.edata[dgl.ETYPE].cpu())
        value_counts = types.value_counts()
        types_to_drop = value_counts[value_counts < minimum_sample_count].index
        print('Ignore DDI types:', sorted(types_to_drop), 'since they have less than', minimum_sample_count, 'samples')
        indices_to_drop = types[types.isin(types_to_drop)].index
        self.ddi_graph = dgl.remove_edges(self.ddi_graph, indices_to_drop)
        self.graph = combine_graphs(split_etype(self.ddi_graph), self.graph['DPI'], self.graph['PDI'], self.graph['PPI'])
        self.graph.ndata['feature'] = self.ndata['feature']
        # self.ddi_graph = self.graph['Drug', :, 'Drug']

        # build model
        out_dim = self.data['DDI']['Y'].nunique() + 1
        self.sage = RGCN(
            in_feats_d = self.data['DrugFeatures'].shape[1],
            in_feats_p = self.data['ProteinFeatures'].shape[1],
            hidden_size = hidden_size,
            out_feats = out_dim,
            rel_names = [f'DDI_{y:02}' for y in self.data['DDI']['Y'].unique()] + ['DPI', 'PDI', 'PPI']
        )
        self.predictor = MLPPredictor(
            in_features = hidden_size,
            out_classes = out_dim
        )
        self.loss_func = nn.CrossEntropyLoss()

    def setup(self, stage: str = None) -> None:
        self.train_eid, self.test_eid = train_test_split(
            np.arange(self.ddi_graph.num_edges()), 
            test_size=0.2, stratify=self.ddi_graph.edata[dgl.ETYPE].cpu()
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DDIDataLoader(self.graph, self.train_eid, batch_size=self.batch_size)

    def training_step(self, batch: tuple, batch_idx: int) -> STEP_OUTPUT:
        input_nodes, pos_pair_graph, neg_pair_graph, graph = batch
        
        h = self.sage(graph.to(self.device), {k: v.to(self.device) for k, v in graph.ndata['feature'].items()})
        pos_score = self.predictor(pos_pair_graph, h['Drug'][pos_pair_graph.ndata[dgl.NID]])
        neg_score = self.predictor(neg_pair_graph, h['Drug'][neg_pair_graph.ndata[dgl.NID]])
        prediction = torch.cat([pos_score, neg_score])
        real_label = torch.cat([pos_pair_graph.edata[dgl.ETYPE] + 1, 
                                torch.zeros(neg_score.shape[0], device=self.device, dtype=torch.int)])
        
        loss = self.loss_func(prediction, real_label)
        self.log('train_loss', loss)
        return loss
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DDIDataLoader(self.graph, self.test_eid, batch_size=self.batch_size)

    def validation_step(self, batch: tuple, batch_idx: int) -> STEP_OUTPUT:
        input_nodes, pos_pair_graph, neg_pair_graph, graph = batch
        
        h = self.sage(graph.to(self.device), {k: v.to(self.device) for k, v in graph.ndata['feature'].items()})
        pos_score = self.predictor(pos_pair_graph, h['Drug'][pos_pair_graph.ndata[dgl.NID]])
        neg_score = self.predictor(neg_pair_graph, h['Drug'][neg_pair_graph.ndata[dgl.NID]])
        prediction = torch.cat([pos_score, neg_score])
        real_label = torch.cat([pos_pair_graph.edata[dgl.ETYPE] + 1, 
                                torch.zeros(neg_score.shape[0], device=self.device, dtype=torch.int)])
        loss = self.loss_func(prediction, real_label)

        pred_label = torch.argmax(prediction, dim=1).cpu()
        real_label = real_label.cpu()
        f1 = f1_score(real_label, pred_label, average='weighted')
        accuracy = accuracy_score(real_label, pred_label)
        precision = precision_score(real_label, pred_label, average='weighted')
        recall = recall_score(real_label, pred_label, average='weighted')
        self.log_dict({'val_f1': f1, 'val_accuracy': accuracy, 'val_precision': precision, 'val_recall': recall, 'val_loss': loss})
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def forward(self, g, features):
        h = self.sage(g, features)
        return self.predictor(g, h['Drug'])


def config_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:

    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand', required=True)

    new_parser = subparsers.add_parser('new', help='Train a new model')
    resume_parser = subparsers.add_parser('resume', help='Resume training a model')

    model_parser = new_parser.add_argument_group('Model', 'Arguments for Model')
    model_parser.add_argument('--pkl-path', type=str, default='../dataset.pkl', help='Path to the dataset pickle file')
    model_parser.add_argument('--hidden-size', type=int, default=DEFAULT_HIDDEN_SIZE, help='Hidden size of the model')
    model_parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate of the optimizer')
    model_parser.add_argument('--batch-size', type=int, default=1024, help='Batch size of the dataloader')
    model_parser.add_argument('--minimum-sample-count', type=int, default=10, help='Minimum sample count of each DDI type')


    resume_parser = resume_parser.add_argument_group('Resume')
    resume_parser.add_argument('checkpoint_path', type=str, default=None)
    
    for p in (model_parser, resume_parser):
        trainer_parser = p.add_argument_group('Trainer', 'Arguments for Trainer')
        trainer_parser.add_argument('--early-stop-patience', type=int, default=3, help='Patience for early stopping')

    return parser

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')
    from rich import traceback, print
    traceback.install()

    parser = config_parser()
    args = parser.parse_args()

    print(args)

    monitor_config = dict(
        monitor='val_loss',
        mode='min',
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath='./model_checkpoint',
        save_top_k=3,
        **monitor_config
    )

    early_stop_callback = EarlyStopping(
        min_delta=0.00,
        patience=args.early_stop_patience,
        **monitor_config
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=args.early_stop_patience,
        verbose=False,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=-1,
        strategy='ddp_find_unused_parameters_true',
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=1,
        profiler='advanced',
    )

    match args.subcommand:
        case 'new':
            model = Model(
                pkl_path=args.pkl_path, 
                hidden_size=args.hidden_size, 
                learning_rate=args.learning_rate,
                minimum_sample_count=args.minimum_sample_count,
                batch_size=args.batch_size
            )
            trainer.fit(model)
        case 'resume':
            print("resume from", args.checkpoint_path)
            model = Model.load_from_checkpoint(args.checkpoint_path)
            trainer.fit(model, checkpoint_path=args.checkpoint_path)