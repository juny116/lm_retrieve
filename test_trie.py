from trie import Trie
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
model = AutoModelForCausalLM.from_pretrained('facebook/bart-large')

sents = [
    "The unicorns greeted the scientists, explaining that they had been expecting the encounter for a while.",
    "The unicorns greeted the scientists, But this is different sentence.",
    "This is the second sentence which is also very long and has a lot of words in it.",
]

sents = [tokenizer.encode(sent)[1:] for sent in sents]
print(sents[0])
prefix_sents = [sent[:5] for sent in sents]

# for sent in sents:
#     print(sent)


trie = Trie(sents)
# for i in range(5,10):
#     print(trie.get(sents[0][:i]))

force_words = ["The"]
force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids
starting_text = [""]

input_ids = tokenizer(starting_text, return_tensors="pt").input_ids

for i in range(10):
    outputs = model.generate(
        input_ids,
        max_new_tokens=1,
        force_words_ids=force_words_ids,
        num_beams=10,
        num_return_sequences=1,
        no_repeat_ngram_size=1,
        remove_invalid_values=True,
    )
    print(outputs[0][2:].tolist())
    print(tokenizer.decode(outputs[0][2:], skip_special_tokens=True))
    force_words_ids = [trie.get(outputs[0][2:].tolist())]
    print(force_words_ids)
    input_ids = outputs


print("Output:\n" + 100 * '-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# print(tokenizer.decode(outputs[1], skip_special_tokens=True))