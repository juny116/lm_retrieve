from seal import FMIndex, fm_index_generate, SEALSearcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PhrasalConstraint


corpus = [
    "Doc 1 @@ Is this science or magic",
    "Doc 2 @@ there are cats and dogs",
    "Doc 3 @@ the scientists were surprised to see the unicorn",
    "Doc 4 @@ I was surprised to see the unicorn",
]
labels = ['doc1', 'doc2', 'doc3', 'doc4']

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large')

def preprocess(doc):
    doc = ' ' + doc
    doc = tokenizer(doc, add_special_tokens=False)['input_ids']
    doc += [tokenizer.eos_token_id]
    return doc

corpus_tokenized = [preprocess(doc) for doc in corpus]

index = FMIndex()
index.initialize(corpus_tokenized, in_memory=True)
index.labels = labels

index.save('legacy/sample_corpus.fm_index')

index = FMIndex.load('legacy/sample_corpus.fm_index')

# constrained generation
query = " ".join("""
The unicorns greeted the scientists, explaining that they had been expecting the encounter for
a while.'
”""".split()).strip()


searcher = SEALSearcher.load('sample_corpus.fm_index', None)
searcher.include_keys = True

for i, doc in enumerate(searcher.search(query, k=3)):
    print(i, doc.score, doc.docid, *doc.text(), sep='\t')
    print("Matched:")
    matched = sorted(doc.keys, reverse=True, key=lambda x:x[2])
    matched = matched[:5]
    for ngram, freq, score in matched:
        print("{:.1f}".format(score).zfill(5), freq, repr(ngram), sep='\t')
