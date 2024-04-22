from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser
)

from dataclasses import dataclass, field
import json
import os

from typing import Optional, List, Dict, Any
import wandb


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )


@dataclass
class DataTrainingArguments:
    test_source: Optional[str] = field(
        default=None,
        metadata={"help": "The test source file."},
    )


@dataclass
class CustomTrainingArguments:
    out_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the output generation file."},
    )


def load_txt_file(
    filepath: str
) -> List[str]:
    data = []
    with open(filepath, 'r', encoding='utf8') as reader:
        lines = reader.readlines()
        for line in lines:
            data.append(line)
    return data


def write_jsonl_file(
    filepath: str,
    input_list: List[Any],
    mode: str ='a+',
    overwrite: bool =False
) -> None:
    if overwrite:
        try:
            os.remove(filepath)
        except:
            pass
    with open(filepath, mode, encoding='utf8') as writer:
        for line in input_list:
            writer.write(json.dumps(line) + '\n')


def process_line(line):
    return line.strip()


def tokenize(
    line: Dict[str, str],
    tokenizer
):
    """Tokenizes the data for sequence to sequence training.

    Args:
        line: The data arguments.
            The data must be stored in jsonl files. Each line of the file must be a string.
            For each train dataset (train/valid) you need:
                - A source file with the input documents (one per line)
                - A target file with the output summaries (one per line)

    Returns:
        tuple: A tuple containing the train dataset and dev dataset.
    """
    tok_input = tokenizer(line['input'], truncation=True, padding='max_length', max_length=1024)
    tok_output = tokenizer(line['output'], truncation=True, padding='max_length', max_length=50)
    return {
        'input_ids': tok_input['input_ids'],
        'attention_mask': tok_input['attention_mask'],
        'labels': tok_output['input_ids'],
    }


def create_datasets(
    data_args,
):
    test_source = list(map(process_line, load_txt_file(data_args.test_source)))
    return test_source


class ModelInit():
    """Class for initializing the model (useful for reproducibility).

    Args:
        model_args (object): Object containing model arguments.
        data_args (object): Object containing data arguments.
        training_args (object): Object containing training arguments.
    """

    def __init__(
        self,
        model_args,
        data_args,
        training_args,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def __call__(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_args.model_name_or_path)
        model.to('cuda')

        return model

    def create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path
        )

        return tokenizer


def create_and_prepare_model(model_args, data_args, training_args):
    model_init = ModelInit(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    model = model_init()
    model.to('cuda')

    tokenizer = model_init.create_tokenizer()

    return model, tokenizer


def generator(model, tokenizer, prompt, max_length):
    tok = tokenizer(prompt, max_length=max_length, padding="max_length", truncate=True, return_tensors='pt').to('cuda')
    gen = model.generate(
        tok['input_ids'],
    )
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


def main(model_args, data_args, training_args):
    # model
    model, tokenizer = create_and_prepare_model(
        model_args, data_args, training_args
    )

    # datasets
    test_source = create_datasets(
        data_args,
    )

    # generate
    outputs = []
    for line in test_source:
        outputs.append(
            generator(model, tokenizer, line, 1024)
        )

    # save
    write_jsonl_file(
        training_args.out_path,
        outputs,
        overwrite=True,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
