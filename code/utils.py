import dgl
import dgl.utils
import torch
import pandas as pd


def num_nodes_dict(graph: dgl.DGLGraph) -> dict[str, int]:
    '''
    Parameters
    ----------
    graph : DGLGraph
        The graph to be converted to a heterograph.

    Returns
    -------
    dict
        The number of nodes for each node type.
    '''
    return {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes}

def to_bidirected(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    '''
    Parameters
    ----------
    graph : DGLGraph
        The graph to be converted to bidirected.
    
    Returns
    -------
    DGLGraph
        The bidirected graph.
    '''
    data = {}
    for (ntype1, etype, ntype2) in graph.canonical_etypes:
        u, v = graph[etype].edges()
        if ntype1 == ntype2:
            u, v = torch.cat([u, v]), torch.cat([v, u])
        data[(ntype1, etype, ntype2)] = (u, v)
    graph_undirected = dgl.heterograph(data, num_nodes_dict=num_nodes_dict(graph))
    for k, v in graph.ndata.items():
        graph_undirected.ndata[k] = v
    for k, v in graph.edata.items():
        graph_undirected.edata[k] = torch.cat([v, v])
    return graph_undirected

def combine_graphs(*graphs, **kwargs) -> dgl.DGLGraph:
    '''
    Parameters
    ----------
    *graphs : DGLGraph
        The graphs to be combined.
    **kwargs : dict
        The arguments passed to dgl.heterograph.

    Returns
    -------
    DGLGraph
        The combined graph. (ndatas and edatas are not preserved)
    '''
    return dgl.heterograph({(n1, e, n2): g[e].edges() for g in graphs for (n1, e, n2) in g.canonical_etypes}, **kwargs)

def split_etype(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    '''
    Parameters
    ----------
    graph : DGLGraph
        The graph to be converted to a heterograph.

    Returns
    -------
    DGLGraph
        The heterograph.
    
    Notes
    -----
    The graph with one edge type 'etype1+etype2+etype3' will be converted to a heterograph with three edge types 'etype1', 'etype2', 'etype3'.
    This function is often used to restore the origin graph from origin_graph[ntype1, : , ntype2].
    '''
    series = pd.Series(graph.edata[dgl.ETYPE])
    data = {}
    for (n1, etypes, n2) in graph.canonical_etypes:
        u, v = graph[etypes].edges()
        for (_, eids), etype in zip(series.groupby(series), etypes.split('+')):
            eids = eids.index
            data[(n1, etype, n2)] = (u[eids], v[eids])
    graph_split = dgl.heterograph(data, num_nodes_dict=num_nodes_dict(graph))
    for k, v in graph.ndata.items():
        if isinstance(v, dict) and len(v) == 1:
            graph_split.ndata[k] = list(v.values())[0]
        else:
            graph_split.ndata[k] = v
    # TODO preserve edata
    return graph_split
    

def get_graph(data: dict[str, pd.DataFrame]) -> dgl.DGLGraph:
    '''
    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        The data to be converted to a heterograph.
        The keys are 'DDI', 'DTI', 'PPI', 'Drugs', 'Proteins', 'DrugFeatures', 'ProteinFeatures'.

    Returns
    -------
    DGLGraph
        The heterograph.
    '''
    graph_data = {
        'DDI': data['DDI'].copy(),
        'DPI': data['DTI'].copy(),
        'PPI': data['PPI'].copy(),
    }
    drug_indices = data['Drugs'].reset_index().reset_index().set_index('Drug_ID')['index']
    protein_indices = data['Proteins'].reset_index().reset_index().set_index('Protein_ID')['index']
    graph_data['DDI']['Drug1_ID'] = data['DDI']['Drug1_ID'].map(drug_indices)
    graph_data['DDI']['Drug2_ID'] = data['DDI']['Drug2_ID'].map(drug_indices)
    graph_data['DPI']['Drug_ID'] = data['DTI']['Drug_ID'].map(drug_indices)
    graph_data['DPI']['Protein_ID'] = data['DTI']['Protein_ID'].map(protein_indices)
    graph_data['PPI']['Protein1_ID'] = data['PPI']['Protein1_ID'].map(protein_indices)
    graph_data['PPI']['Protein2_ID'] = data['PPI']['Protein2_ID'].map(protein_indices)

    # Create a heterograph from the graph data
    g = dgl.heterograph({
        ('Drug', f'DDI_{y:02}', 'Drug'): (
            torch.tensor(d['Drug1_ID'].values),
            torch.tensor(d['Drug2_ID'].values)
        ) for y, d in graph_data['DDI'].groupby('Y') 
    } | {
        ('Drug', 'DPI', 'Protein'): (
            torch.tensor(graph_data['DPI']['Drug_ID'].values), 
            torch.tensor(graph_data['DPI']['Protein_ID'].values)
        ),
        ('Protein', 'PDI', 'Drug'): (
            torch.tensor(graph_data['DPI']['Protein_ID'].values), 
            torch.tensor(graph_data['DPI']['Drug_ID'].values)
        ),
        ('Protein', 'PPI', 'Protein'): (
            torch.tensor(graph_data['PPI']['Protein1_ID'].values),
            torch.tensor(graph_data['PPI']['Protein2_ID'].values)
        )
    })
    g.nodes['Drug'].data['feature'] = torch.tensor(data['DrugFeatures'])
    g.nodes['Protein'].data['feature'] = torch.tensor(data['ProteinFeatures'])
    return g

