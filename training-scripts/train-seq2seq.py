from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset

import os
import glob
import importlib.util
folder_path = "/content/utils/scripts"
py_files = glob.glob(os.path.join(folder_path, "*.py"))
for py_file in py_files:
    module_name = os.path.splitext(os.path.basename(py_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update({name: getattr(module, name) for name in dir(module) if callable(getattr(module, name))})


model_name_or_path = "facebook/bart-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Train
def process_sources(line):
    return line.strip()
def process_targets(line):
    keys = line.split(' ')
    keys = list(filter(lambda x: x not in [' ', '|'], keys))
    return ' '.join(keys).strip()
sources = list(map(process_sources, load_txt_file('/content/ctrl-sum/datasets/cnndm/train.source')))
targets = list(map(process_targets, load_txt_file('/content/ctrl-sum/datasets/cnndm/train.oraclewordns')))
data_dict = {'input': sources[:100], 'output': targets[:100]}
train_dataset = Dataset.from_dict(data_dict)
def tokenize(line):
    tok_input = tokenizer(line['input'], truncation=True, padding='max_length', max_length=1024)
    tok_output = tokenizer(line['output'], truncation=True, padding='max_length', max_length=50)
    return {
        'input_ids': tok_input['input_ids'],
        'attention_mask': tok_input['attention_mask'],
        'labels': tok_output['input_ids'],
    }
train_dataset = train_dataset.map(lambda data: tokenize(data), batched=True)
train_dataset = train_dataset.remove_columns(['input', 'output'])

# Valid
sources = list(map(process_sources, load_txt_file('/content/ctrl-sum/datasets/cnndm/val.source')))
targets = list(map(process_targets, load_txt_file('/content/ctrl-sum/datasets/cnndm/val.oraclewordns')))
data_dict = {'input': sources[:100], 'output': targets[:100]}
eval_dataset = Dataset.from_dict(data_dict)
def tokenize(line):
    tok_input = tokenizer(line['input'], truncation=True, padding='max_length', max_length=1024)
    tok_output = tokenizer(line['output'], truncation=True, padding='max_length', max_length=50)
    return {
        'input_ids': tok_input['input_ids'],
        'attention_mask': tok_input['attention_mask'],
        'labels': tok_output['input_ids'],
    }
eval_dataset = eval_dataset.map(lambda data: tokenize(data), batched=True)
eval_dataset = eval_dataset.remove_columns(['input', 'output'])

# Define training arguments (adjust as needed)
training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    save_steps=2000,
    save_total_limit=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8
)

# # Collator
# data_collator = DataCollatorForSeq2Seq(
#     tokenizer=tokenizer,
#     model=model,
#     padding='max_length',
#     max_length=1024,
#     # return_tensors
# )

# Create the Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()