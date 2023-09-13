import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, EsmTokenizer, PreTrainedTokenizer
from tqdm import tqdm

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to_device(d: dict[str, torch.Tensor]):
        return {k: v.to(device) for k, v in d.items()}
    
    def pipeline(model: AutoModel, tokenizer: PreTrainedTokenizer):
        model = model.to(device)
        model.eval()
        def _pipeline(text: str):
            tokenized = tokenizer(text, return_tensors='pt', max_length=512, padding=True, truncation=True)
            return model(**to_device(tokenized)).pooler_output[0].cpu().numpy()
        return _pipeline

    data = {'Proteins': pd.read_csv("../data/csv/Proteins.csv").set_index('Protein_ID'),
            'Drugs': pd.read_csv("../data/csv/Drugs.csv").set_index('Drug_ID')}
    
    ChemBert = pipeline(model=AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM"),
                        tokenizer=AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM"))

    Esm1b = pipeline(model=AutoModel.from_pretrained("facebook/esm-1b"), 
                     tokenizer=EsmTokenizer.from_pretrained('facebook/esm-1b', do_lower_case=False))

    data['DrugFeatures'] = np.array([ChemBert(drug) for drug in tqdm(data['Drugs']['Drug'], desc='DrugFeatures')])
    data['ProteinFeatures'] = np.array([Esm1b(protein)for protein in tqdm(data['Proteins']['Protein_seq'], desc='ProteinFeatures')])

    np.save("../data/npy/DrugFeatures.npy", data['DrugFeatures'])
    np.save("../data/npy/ProteinFeatures.npy", data['ProteinFeatures'])