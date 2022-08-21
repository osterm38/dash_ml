"""
A module to help with simple local caching (loading from/writing to file).

"""
# IMPORTS
from dataclasses import dataclass
import datasets as dx
from pathlib import Path
import transformers as tx
from typing import Dict, List, Optional, Union
from .utils import get_logger

# GLOBAL VARS
# TODO: figure a way to dynamic change log level?
LOG = get_logger(name=__name__, level='DEBUG')
CACHE_BASE = Path.home() / '.cache' / 'ml_utils' # TODO: make dynamic using os.environ['ML_UTILS_CACHE_BASE']?
FILE_SUFFIXES = ['csv', 'json']
MODEL_NAMES = [
    #'text-generation'
    "facebook/opt-125m", # 2g of RAM, 1.2ba/s with ba=10, #tokens=30-60 (so about 10-15 small sentences per sec)
    "facebook/opt-350m",
    "facebook/opt-1.3b", # 2.5g download, 8g of RAM just for model
    # "facebook/opt-2.7b",
]

# FUNCTIONS
def load_file_as_dataset(
    path: Union[str, Path],
    type_: Optional[str] = None,
) -> dx.Dataset:
    """given a file path (optionally hinting it's suffix), load it as a datasets.Dataset"""
    LOG.debug(f'inputs: {path=}, {type_=}')
    
    # path
    path = Path(path).resolve()
    assert path.is_file(), f'oops, {path=} should exist as a file'
    # type_
    if type_ is None:
        type_ = path.suffix
    type_ = type_.replace('.', '')
    assert type_ in FILE_SUFFIXES, f'oops, {type_=} should be in {FILE_SUFFIXES=}'

    LOG.debug(f'cleaned: {path=}, {type_=}')

    # load
    ds = dx.load_dataset('csv', data_files=str(path), split='train')
    LOG.debug(f'loaded: {ds=}')
    
    return ds


def load_files_as_datasetdict(
    paths: Union[List[Union[str, Path]], Dict[str, Union[str, Path]]],
    type_: Optional[str] = None,
) -> dx.DatasetDict:
    """given a list or dict of paths, load each and combine to a datasets.DatasetDict (keys default to index order of list, unless dict given)"""
    LOG.debug(f'inputs: {paths=}, {type_=}')
    if isinstance(paths, list):
        paths = dict(enumerate(paths))
    
    dsd = {}
    for key, path in paths.items():
        LOG.debug(f'loading: {key=}')
        ds = load_file_as_dataset(path=path, type_=type_)
        dsd[key] = ds
    dsd = dx.DatasetDict(dsd)
    LOG.debug(f'loaded: {dsd=}')
    
    return dsd


def load_hf_named_dataset(
    name: str,
) -> Union[dx.Dataset, dx.DatasetDict]:
    """given a named dataset (from just huggingface for now), load it"""
    # TODO: can we name our datasets for caching?
    ds = dx.load_dataset(name)
    return ds


# CLASSES
class Loader:
    CACHE = CACHE_BASE
    
    @classmethod
    def get_output_path(self, name: str) -> Path:
        self.CACHE.mkdir(parents=True, exist_ok=True)
        return self.CACHE / self.get_clean_name(name)
        
    @classmethod
    def get_clean_name(self, name: str) -> str:
        """clean stem name of a path so it only adds single subdirectory.
        e.g.  s = 'a/b/c.csv' -> 'a_b_c'
        """
        return name.replace('/', '_').split('.')[0]
    
    @classmethod
    def name_exists(self, name: str) -> bool:
        return self.get_output_path(name).is_dir()
    

class DatasetDictLoader(Loader):
    """a wrapper around datasets.DatasetDict for convenience where we need train/validation/test split (train must be nonempty)"""

    DSD_KEYS = ['train', 'validation', 'test']
    DSD_FEATURES = [] # inherit/redefine in subclasses?
    CACHE = Loader.CACHE / 'datasets'

    @classmethod
    def load(self, name: str, overwrite: bool = False, paths: Optional[Union[str, Path, Dict[str, Union[str, Path]]]] = None) -> dx.DatasetDict:
        """loads a datasetdict either from file (if nonexistent) or from cache"""
        # TODO: come up with default way to populate a name?
        LOG.debug(f'inputs: {name=}, {overwrite=}, {paths=}, {self.CACHE=}, {self.DSD_KEYS=}, {self.DSD_FEATURES=}')
        if paths is None and not self.name_exists(name):
            return load_hf_named_dataset(name)
        # else
        out_path = self.get_output_path(name)
        if (overwrite or not self.name_exists(name)) and paths is not None:
            dsd = self.load_new(paths=paths)
            dsd.save_to_disk(str(out_path))
        assert self.name_exists(name), f'oops, named dataset doesnt exist, i.e. {out_path=}?'
        dsd = dx.load_from_disk(str(out_path))
        return dsd
  
    @classmethod
    def load_new(self, paths: Union[str, Path, Dict[str, Union[str, Path]]]) -> dx.DatasetDict:
        """loads a new datasetdict from file(s)"""
        # TODO: for now, this loads the same way each time called, but eventually need to name/cache/reload from cache 2nd time around?!
        if isinstance(paths, str) or isinstance(paths, Path):
            paths = {'train': paths}
        assert 'train' in paths, f'oops, need at least one dataset with key "train"'
        dsd = load_files_as_datasetdict(paths)
        train_features = dsd['train'].features
        # check keys and features present where needed
        LOG.debug(f'{self.__class__.__name__}: {self.DSD_KEYS=}, {self.DSD_FEATURES=}')
        for k in self.DSD_KEYS:
            if k not in dsd.keys():
                dummy_dct = {f: [] for f in train_features}
                dsd[k] = dx.Dataset.from_dict(dummy_dct)
            ds = dsd[k]
            for f in self.DSD_FEATURES:
                assert f in ds.features, f'oops, need "text" to be in {k=}:{ds.features=}'
                if 'label' not in ds.features:
                    LOG.warning(f'"label" not a feature in {train_features=}, ok for now')
        # TODO: cache?
        return dsd


class TextLoader(DatasetDictLoader):
    """a raw dataset wrapper where we only need a text column"""
    DSD_FEATURES = DatasetDictLoader.DSD_FEATURES + ['text']
    CACHE = DatasetDictLoader.CACHE / 'texts'
   
        
class EmbeddedTextLoader(TextLoader):
    """a raw dataset wrapper where we need embedding columns CLS/MAX/MEAN"""
    DSD_FEATURES = TextLoader.DSD_FEATURES + ['CLS', 'MAX', 'MEAN']
    CACHE = TextLoader.CACHE.parent / 'embeddings'
    

class ModelLoader(Loader):
    """A simple loader for (transformers-based) sequence classification models"""
    CACHE = Loader.CACHE / 'models'
    MOD_TYPE = tx.AutoModelForSequenceClassification
    
    @classmethod
    def load(self, name: str):
        LOG.debug(f'inputs: {name=}, {self.MOD_TYPE=}')
        if self.name_exists(name):
            name = self.get_output_path(name)
            LOG.debug(f'loading local model')
        else:
            assert name in MODEL_NAMES, f'{name=} is restricted to {MODEL_NAMES=}'
            LOG.debug(f'loading downloaded model')
        LOG.debug(f'{name=}')
        # assumes config/tokenizer/model together!
        config = tx.AutoConfig.from_pretrained(name)
        tokenizer = tx.AutoTokenizer.from_pretrained(name)
        model = self.MOD_TYPE.from_pretrained(name, config=config)
        return ModelTokenizerPair(model, tokenizer)
    

@dataclass
class ModelTokenizerPair:
    model: tx.AutoModel
    tokenizer: tx.AutoTokenizer
