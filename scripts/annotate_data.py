import json
from transformers import DistilBertTokenizerFast

# Load annotated data
with open(r'C:\Users\oleja\PycharmProjects\NLP test\data/annotated_data.json', 'r') as f:
    annotated_data = json.load(f)

# Load pre-trained tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Tokenize a few examples manually
for item in annotated_data[:2]:  # Limit to first 2 examples for brevity
    text = item['text']
    entities = item['entities']

    # Tokenize text
    tokenized_input = tokenizer(text, truncation=True, padding='max_length', is_split_into_words=False)
    word_ids = tokenized_input.word_ids()

    print(f"Text: {text}")
    print(f"Tokens: {tokenized_input.tokens()}")
    print(f"Word IDs: {word_ids}")
    print(f"Entities: {entities}")

    # Align labels manually
    labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            entity = next((e for e in entities if e['start'] <= word_idx < e['end']), None)
            labels.append(entity['label'] if entity else 0)
        else:
            labels.append(-100)
        previous_word_idx = word_idx

    print(f"Labels: {labels}")
    print("\n---\n")
