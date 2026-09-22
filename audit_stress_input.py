"""Independent all-row PAWS encoding audit before any stress inference."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from tokenization import load_tokenizer

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--asset-dir',type=Path,required=True);p.add_argument('--paws-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    frame=pd.read_parquet(a.paws_dir/'test.parquet')
    if len(frame)!=8000 or frame.id.nunique()!=8000 or set(frame.label)!={0,1}:
        raise ValueError('Bad PAWS schema or IDs')
    pairs=list(zip(frame.sentence1,frame.sentence2))
    raw=Tokenizer.from_file(str(a.asset_dir/'model/tokenizer.json'))
    lengths=np.array([len(e.ids) for e in raw.encode_batch(pairs)])
    raw.enable_padding(length=128,pad_id=1,pad_token='<pad>');raw.enable_truncation(max_length=128)
    independent=raw.encode_batch(pairs)
    correct=load_tokenizer(a.asset_dir/'model')(frame.sentence1.tolist(),frame.sentence2.tolist(),truncation=True,max_length=128,padding='max_length',return_tensors='np')
    for key,attribute in [('input_ids','ids'),('attention_mask','attention_mask')]:
        if not np.array_equal(correct[key],np.array([getattr(e,attribute) for e in independent])):
            raise ValueError('PAWS encoding mismatch: '+key)
    def normalized_pair(s1,s2):
        return tuple(sorted((' '.join(s1.lower().split()),' '.join(s2.lower().split()))))
    paws_keys={normalized_pair(x,y) for x,y in pairs}
    overlaps={}
    for split in ['train','validation']:
        f=pd.read_parquet(a.asset_dir/'data'/f'{split}-00000-of-00001.parquet')
        overlaps[split]=len(paws_keys & {normalized_pair(r.sentence1,r.sentence2) for r in f.itertuples()})
    result={'status':'PASS','rows':len(frame),'unique_source_ids':int(frame.id.nunique()),'ids_are_source_id_not_dataframe_row_index':True,'raw_asset_and_transformers_all_row_input_ids_and_masks_equal':True,'untruncated_token_lengths':{'minimum':int(lengths.min()),'median':float(np.median(lengths)),'maximum':int(lengths.max()),'over_128_count':int((lengths>128).sum())},'normalized_unordered_exact_pair_overlap_with_mrpc':overlaps,'label_counts':{str(k):int(v) for k,v in frame.label.value_counts().items()},'data_sha256':hashlib.sha256((a.paws_dir/'test.parquet').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Equality checks preprocessing only; PAWS labels never select MRPC checkpoints.'}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
