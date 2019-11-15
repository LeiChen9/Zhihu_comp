import os 
import random 
import msgpack 
from .utils.vocab import Vocab, Indexer
from .utils.loader import load_data, load_embeddings

class Interface:
    """
    interface = Interface(self.args, self.log)

    Attrs:
        args
        target_map: w2id & id2w
        vocab
        num_classes
        num_vocab
        padding

    Funcs:
        
        __init__:
            load args
            build/load vocab and target map

        load_embeddings

        pre_process: filter text < max_len, sort by len, make_batch

        process_sample: 
            return processed
                processed['text']: id-like words
                processed['len']
                processed['target']

        shuffle_batch

        make_batch

        padding: pad 0

        post_process: turn prob to idx. Return final_prediction
            
    """
    def __init__(self, args, log=None):
        self.args = args 
        # build/load vocab and target map
        vocab_file = os.path.join(args.output_dir, 'vocab.txt')  # args.output_dir looks like: 'models/snli'
        target_map_file = os.path.join(args.output_dir, 'target_map.txt')
        if not os.path.exists(vocab_file):
            data = load_data(self.args.data_dir)
            """
            data looks like:
                {'text1': 'this church choir sings to the masses as they sing joyous songs from the book at a church', 
                'text2': 'the church has cracks in the ceiling', 
                'target': '1'}
            """
            self.target_map = Indexer.build((sample['target'] for sample in data), log=log)
            """
            target_map attr looks like:
                w2id:{'0': 0, '1': 1, '2': 2}
                id2w:{0: '0', 1: '1', 2: '2'}
            """
            self.target_map.save(target_map_file)
            self.vocab = Vocab.build((word for sample in data
                                        for text in (sample['text1'], sample['text2'])
                                        for word in text.split()[:self.args.max_len]),
                                        lower=args.lower_case, min_df=self.args.min_df, log=log,
                                        pretrained_embeddings=args.pretrained_embeddings,
                                        dump_filtered=os.path.join(args.output_dir, 'filtered_words.txt'))
            # import pdb; pdb.set_trace()
            self.vocab.save(vocab_file)
        else:
            self.target_map = Indexer.load(target_map_file)
            self.vocab = Vocab.load(vocab_file)
        args.num_classes = len(self.target_map)
        args.num_vocab = len(self.vocab)
        args.padding = Vocab.pad()  # 0
        

    def load_embeddings(self):
        '''generate embeddings suited for the current vocab or load previously cached ones.'''
        embedding_file = os.path.join(self.args.output_dir, 'embedding.msgpack')
        if not os.path.exists(embedding_file):
            embeddings = load_embeddings(self.args.pretrained_embeddings, self.vocab,
                                            self.args.embedding_dim, mode=self.args.embedding_mode,
                                            lower=self.args.lower_case)
            with open(embedding_file, 'wb') as f:
                msgpack.dump(embeddings, f)
        else:
            with open(embedding_file, 'rb') as f:
                embeddings = msgpack.load(f)
        return embeddings
    
    def pre_process(self, data, training=True):
        result = [self.pre_process(sample) for sample in data]
        if training:
            result = list(filter(lambda x: x['len1'] < self.args.max_len and x['len2'] < self.args.max_len, result))
            if not self.args.sort_by_len:
                return result 
            result = sorted(result, key=lambda x: (x['len1'], x['len2'], x['text1']))
        batch_size = self.args.batch_size
        return [self.make_batch(result[i:i+batch_size]) for i in range(0, len(data), batch_size)]
    
    def process_sample(self, sample, with_target=True):
        '''
        Intro:
        Args:
            sample:
            with_target:
        
        Returns:
            

        '''
        text1 = sample['text1']
        text2 = sample['text2']
        if self.args.lower_case:
            text1 = text1.lower()
            text2 = text2.lower()
        processed = {
            'text1': [self.vocab.index(w) for w in text1.split()[:self.args.max_len]],
            'text2': [self.vocab.index(w) for w in text2.split()[:self.args.max_len]]
        }
        processed['len1'] = len(processed['text1'])
        processed['len2'] = len(processed['text2'])
        if 'target' in sample and with_target:
            target = sample['target']
            assert target in self.target_map
            processed['target'] = self.target_map.index(target)
        return processed

    def shuffle_batch(self, data):
        data = random.sample(data, len(data))
        if self.args.sort_by_len:
            return data 
        batch_size = self.args.batch_size
        batches = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]
        return list(map(self.make_batch, batches))
    
    def make_batch(self, batch, with_target=True):
        batch = {key: [sample[key] for sample in batch] for key in batch[0].keys()}
        if 'target' in batch and not with_target:
            del batch['target']
        batch = {key: self.padding(value, min_len=self.args.min_len) if key.startswith('text') else value 
                    for key, value in batch.items()}
        return batch 
    
    @staticmethod
    def padding(samples, min_len=1):
        max_len = max(max(map(len, samples)), min_len)
        batch = [sample + [Vocab.pad()] * (max_len - len(sample)) for sample in samples]
        return batch
    
    def post_process(self, output):
        final_prediction = []
        for prob in output:
            idx = max(range(len(prob)), key=prob.__getitem__)
            target = self.target_map[idx]
            final_prediction.append(target)
        return final_prediction
