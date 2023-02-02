from trie import Trie
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

sents = [
    "The unicorns greeted the scientists, explaining that they had been expecting the encounter for a while.",
    "The unicorns greeted the scientists, But this is different sentence.",
    "This is the second sentence which is also very long and has a lot of words in it.",
]

sents = [tokenizer.encode(sent) for sent in sents]
prefix_sents = [sent[:5] for sent in sents]

for sent in sents:
    print(sent)

trie = Trie(sents)
for i in range(5,10):
    print(trie.get(sents[0][:i]))

