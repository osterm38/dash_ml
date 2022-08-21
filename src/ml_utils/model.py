
class Model:
    """sklearn-like model (API-wise) for my classification bert models?"""
    def __init__(self):
        # store params
        # self.model_ = None
        pass
    
    def fit(self, X, y=None):
        # use run_glue and follow do_fit
        model = self.load_model()
        # self.model_ = model
        trainer = ...
        trainer.train()
        trainer.save()
        return self
    
    def predict(self, X):
        # use evaluate, break into procs 1-1 with gpus
        model = self.load_model()
        # self.model_ = model
        y_pred = ...
        return y_pred
    
    def load_model(self):
        model = ...
        return model

def serve_up():
    """
    - choose model (dropdown)
      - host/await in gpu
    - show 3D embedding space (plot)
      - on standby
    - update (text) data (upload, new data)
      - runs through predictions/embeddings
        - cache/checksum each, since expensive?
      - (re)train a x-dim -> y-dim embedding
      - a.(re)train a y-dim -> 3-dim
        - (re)create dataframe with xyz+hovering/coloring options
        - change colorings (dropdown)
        - store this
      - b.perform y-dim topic extraction/salience something or other?
        - tx2 borrowings?
        - bertopic?
    """

def main():
    # instantiate model
    m = Model() # or any other model/pipeline from sklearn

    # read in data
    ds = ...
    # format data
    X, y = ds, None

    # a. train up the model
    m = m.fit(X, y)
    # b. and/or predict with model
    y_pred = m.predict(X)
