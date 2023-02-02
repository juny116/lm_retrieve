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

#### FM index const gen ####
out = fm_index_generate(
    model, index,
    **tokenizer([' ' + query], return_tensors='pt'),
    keep_history=False,
    transformers_output=True,
    always_allow_eos=True,
    max_length=100,
)

print(tokenizer.decode(out[0], skip_special_tokens=True).strip())


#### Huggingface const gen ####
encoder_input_str = "The unicorns greeted the scientists, explaining that they had been expecting the encounter for a while."

constraints = [
    PhrasalConstraint(
        tokenizer("surprised", add_special_tokens=False).input_ids
    )
]

input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


outputs = model.generate(
    input_ids,
    constraints=constraints,
    num_beams=10,
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
    max_new_tokens=100,
)


print("Output:\n" + 100 * '-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))