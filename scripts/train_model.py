import json
import logging
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, Features, Sequence, Value

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Load annotated data
with open(r'C:\Users\oleja\PycharmProjects\NLP test\data/annotated_data.json', 'r') as f:
    annotated_data = json.load(f)

# Convert annotated data to the format expected by the `datasets` library
texts = [example['text'] for example in annotated_data]
entities = [example['entities'] for example in annotated_data]

dataset_dict = {
    'text': texts,
    'entities': entities,
}

# Create Dataset object
features = Features({
    'text': Value('string'),
    'entities': Sequence({
        'start': Value('int32'),
        'end': Value('int32'),
        'label': Value('string'),
    })
})

dataset = Dataset.from_dict(dataset_dict, features=features)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', is_split_into_words=False)
    labels = []

    for i, word_ids in enumerate(tokenized_inputs.word_ids(batch_index=0)):
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                if examples['entities'] and len(examples['entities']) > i:
                    entity = next((e for e in examples['entities'][i] if e['start'] <= word_idx < e['end']), None)
                    label_ids.append(entity['label'] if entity else 0)
                else:
                    label_ids.append(0)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

logger.info("Tokenizing dataset...")

try:
    # Add detailed logging before mapping
    logger.info(f"Dataset before tokenization: {dataset}")

    # Log the first few examples for inspection
    for idx, example in enumerate(dataset):
        if idx < 5:  # Log only the first 5 examples
            logger.info(f"Example {idx}: {example}")

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    # Log the tokenized dataset
    logger.info(f"Tokenized dataset: {tokenized_dataset}")

except Exception as e:
    logger.error(f"Error during tokenization: {e}")
    logger.error("Tokenization failed. Training is aborted.")
    exit(1)

# Load pre-trained model
num_labels = len(set(label for example in annotated_data for entity in example['entities'] for label in entity['label']))
model = DistilBertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
logger.info("Starting training...")
trainer.train()

logger.info("Training finished.")
