"""
A module to help with large language model usage (e.g. embedding/predicting).

"""
# IMPORTS
import datasets as dx
import multiprocessing as mp
from pathlib import Path
import torch
import transformers as tx
from typing import Optional, List, Any
from .caching import ModelLoader, ModelTokenizerPair
from .utils import get_logger

# GLOBAL VARS
LOG = get_logger(name=__name__, level='DEBUG')


# CLASSES
class Embedder:
    """A (transformers-based) large language model wrapper that enables extracting embeddings easily/efficiently."""
    def __init__(self, name: str):
        LOG.debug(f'inputs: {name=}')
        self.model_name = name # save for on-the-fly initialization of model
        self.batch_size = 10 # size of chunk sent to a model
        self.num_proc = 1 # ~num gpus, or num models instantiated in parallel
        # TODO: assert/clamp args?, especially with use_cuda?
        
    def _load_model(self) -> ModelTokenizerPair:
        return ModelLoader.load(self.model_name)
        
    def embed(self, ds: dx.DatasetDict, concat: bool = False) -> dx.DatasetDict:
        """run ds through model and grab (possibly concatenate) CLS/MAX/MEAN embeddings, returning enlarged datasetdict"""
        if self.num_proc <= 1: # don't spawn parallel procs, stick with this one
            # _embed_single_process
            res = self._embed_single_process(ds, concat)
        else: # self.num_proc > 1
            res = self._embed_chunked(ds, concat)
        LOG.debug(f'{res=}')
        return res
    
    def _embed_chunked(self, ds: dx.DatasetDict, concat: bool) -> ...:
        # TODO: this should be called in each split of embed into manual subprocs, 1 per gpu
        # for x ... _embed_single_process()
        mp.set_start_method('spawn')
        queue = mp.Queue()
        # kickstart parallel procs
        procs = []
        for i in range(self.num_proc):
            ranges = {k: range(int(len(d) * i/ self.num_proc), int(len(d) * (i+1)/ self.num_proc)) for k, d in ds.items()}
            LOG.debug(f'{i=}; {ranges=}')
            proc = mp.Process(
                target=self._embed_single_process,
                kwargs=dict(
                    ds=dx.DatasetDict({k: d.select(ranges[k]) for k, d in ds.items()}),
                    concat=concat,
                    rank=i,
                    queue=queue,
                ),
            )
            proc.start()
            LOG.debug(f'proc {i=} of {self.num_proc=} started')
            procs.append(proc)
        # await a single response from each
        res = []
        for i, _ in enumerate(procs):
            rank_ds = queue.get() # returns (rank, ds)
            res.append(rank_ds)
            LOG.debug(f'awaiting {i}th result (of {self.num_proc}')
        # double check we've joined all procs
        for i, p in enumerate(procs):
            p.join()
            LOG.debug(f'joined proc {i=} (of {self.num_proc})')
        # sort by rank, omitting it from final list to concat (to keep order of data same as input)
        res = list(zip(*sorted(res)))[1]
        res = {k: [d[k] for d in res] for k in ds.keys()}
        res = dx.DatasetDict({k: dx.concatenate_datasets(vs) for k, vs in res.items()})        
        return res

    def _embed_single_process(self, ds: dx.DatasetDict, concat: bool, rank: Optional[int] = None, queue: Optional[mp.Queue] = None):
        # called in a single process, so we can send model/results/computations through gpu
        mod_tok = self._load_model()
        res_ds = ds.map(
            self._embed_texts,
            input_columns='text',
            batched=True,
            batch_size=self.batch_size,
            fn_kwargs=dict(
                model=mod_tok.model.to('cpu'), # TODO: update with device/gpu eventually?
                tokenizer=mod_tok.tokenizer,
            ),
        )
        if concat:
            res_ds = dx.DatasetDict({k: dx.concatenate_datasets([ds[k].remove_columns('text'), d], axis=1) for k, d in res_ds.items()})
        if rank is not None and queue is not None:
            queue.put((rank, res_ds))
        return res_ds
        
    def _embed_texts(self, texts: List[str], model: tx.AutoModel, tokenizer: tx.AutoTokenizer) -> Any:
        # b: batch size, n: max num tokens, v: vocab size, m: hidden dim
        # tokenized input_ids = (b, n)
        # model logits = (b, n, v) # vocab size!
        # model hidden_states = 13 x (b, n, m)
        # return (b, m)
        # # first tokenize
        tok = tokenizer(texts, truncation=False, padding=True, return_tensors='pt')
        with torch.no_grad():
            mod = model(**tok, output_hidden_states=True)
        mask = tok.attention_mask.unsqueeze(-1) # (b, n, 1)
        hidd = mod.hidden_states[-1] # (b, n, m)
        assert mask.shape[:-1] == hidd.shape[:-1]
        hidd.masked_fill_(mask == 0, float('nan'))
        # LOG.debug(f'{hidd.shape=}')
        # TODO: what about probabilities/labels???
        res = {
            'MEAN': hidd.nanmean(dim=1).cpu().numpy(),
            # 'MEDIAN': hidd.nanmedian(dim=1).cpu().numpy(),
            'CLS': hidd.select(dim=1, index=0).cpu().numpy(), # assume no nans?
            'MAX': hidd.masked_fill(mask == 0, float('-inf')).amax(dim=1).cpu().numpy(),
        }
        return res
