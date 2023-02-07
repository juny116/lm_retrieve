from trie import Trie
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')

sents = [
    "The unicorns greeted the scientists with gratitude.",
    "The unicorns greeted the scientists, explaining that they had been expecting the encounter for a while.",
    "The unicorns greeted the scientists, But this is different sentence.",
    "This is the second sentence which is also very long and has a lot of words in it.",
]

sents = [tokenizer.encode(sent) for sent in sents]
print(sents[0])
print(sents[1])
# for sent in sents:
#     print(sent)


trie = Trie(sents)
# for i in range(5,10):
#     print(trie.get(sents[0][:i]))

force_words = ["The"]
force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids
starting_text = [""]

input_ids = tokenizer(starting_text, return_tensors="pt").input_ids
print(force_words_ids)

for i in range(100):
    outputs = model.generate(
        input_ids,
        max_new_tokens=1,
        force_words_ids=force_words_ids,
        num_beams=200,
        num_return_sequences=1,
        remove_invalid_values=True,
    )
    # print(outputs[0][1:].tolist())
    # print(tokenizer.decode(outputs[0][1:], skip_special_tokens=True))
    input_ids = torch.cat((input_ids[0], outputs[0][1:]), dim=0).expand(1, -1)
    force_words_ids = [trie.get(input_ids[0][1:].tolist())]
    # print(force_words_ids)
    if len(force_words_ids[0]) > 1:
        force_words_ids = [[[ids] for ids in force_words_ids[0]]]
    if not force_words_ids[0]:
        break
    # print(force_words_ids)
    # input_ids = input_ids + outputs


print("Output:\n" + 100 * '-')
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
# print(tokenizer.decode(outputs[1], skip_special_tokens=True))