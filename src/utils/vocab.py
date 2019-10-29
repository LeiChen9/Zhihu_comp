from collections import Counter

class Indexer:
    def __init__(self):
        self.w2id = {}
        self.id2w = {}
    
    @property
    def n_spec(self):
        return 0
    
    def __len__(self):  # make a object like a list. Like len(Indexer)
        return len(self.w2id)
    
    def __getitem__(self, index):
        if index not in self.id2w:
            raise IndexError(f'invalid index {index} in indices.')
        return self.id2w[index]
        