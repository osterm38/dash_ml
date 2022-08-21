# IMPORTS
from pathlib import Path
from dash_ml.utils import get_logger
from dash_ml.caching import TextLoader, EmbeddedTextLoader, ModelLoader
from dash_ml.model import Embedder

# GLOBAL VARS
# TODO: figure a way to dynamic change log level?
LOG = get_logger(name=__name__, level='DEBUG')
HERE = Path(__file__).parent


def main():
    LOG.debug('starting...')
    LOG.debug(f'{HERE=}')
    
    # user specified project name, model name, and path(s) to raw texts
    proj_name = 'rotten_tomatoes' 
    text_paths = None
    model_name = 'facebook/opt-125m'
    
    # proj_name = 'mytrain1'
    # text_paths ='train1.csv'
    # model_name = 'facebook/opt-125m'

    # texts (first time run, will cache our model for easier reloading from TextLoader)
    LOG.debug(f'{proj_name=}, {text_paths=}, {model_name=}')
    text_ds = TextLoader.load(name=proj_name, paths=text_paths, overwrite=False)
    LOG.debug(f'{text_ds=}')
    # embed with specific model (points to local model name?)
    # model = ModelLoader.load(name=model_name, path=...)
    # embedder might run modelloader internally
    # could do this, if rather than getting raw text_paths, we get embedded text_paths, i.e. they have MAX/CLS/MEAN features
    # this happens when we run embeddings outside this and save to files and we want to use this script to cache for easy reloading
    # emb_ds = EmbeddedTextLoader.load(name=proj_name, paths=text_paths, overwrite=False)
    # assert ModelLoader.name_exists(model_name), f'oops, {model_name=} should exist'
    embedder = Embedder(model_name) # load and store this somehwere in dash app
    emb_name = f'{proj_name}/{model_name}'
    LOG.debug(f'{emb_name=}')
    if not EmbeddedTextLoader.name_exists(emb_name):
        emb_path = EmbeddedTextLoader.get_output_path(emb_name)
        LOG.debug(f'{emb_path=}')
        # will be most computational part when using LLMs!
        emb_ds = embedder.embed(text_ds, concat=True)
        emb_ds.save_to_disk(str(emb_path))
    # either way load for consistency
    emb_ds = EmbeddedTextLoader.load(name=emb_name, overwrite=False)
    LOG.debug(f'{emb_ds=}')
    
    # finally, start projecting/visualizing

    LOG.debug('...ending')


if __name__ == "__main__":
    main()