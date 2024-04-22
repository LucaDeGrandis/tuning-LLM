from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser
)

from dataclasses import dataclass, field
import json

from typing import Optional, List, Dict
import wandb


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    wandb_project: str = field(
        default=None,
        metadata={
            "help": "The wandb project name."
        }
    )
    wandb_run_name: str = field(
        default=None,
        metadata={
            "help": "The wandb run name."
        }
    )


@dataclass
class DataTrainingArguments:
    train_source: Optional[str] = field(
        default=False,
        metadata={"help": "The training source file."},
    )
    train_target: Optional[str] = field(
        default=False,
        metadata={"help": "The training target file."},
    )
    dev_source: Optional[str] = field(
        default=False,
        metadata={"help": "The dev source file."},
    )
    dev_target: Optional[str] = field(
        default=False,
        metadata={"help": "The dev target file."},
    )


@dataclass
class CustomTrainingArguments:
    overwrite_output_dir: Optional[bool] = field(
        default=False,
        metadata={"help": "Wheter to overwrite the output directory. Useful for continuing training."},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Number of training epochs for taining."},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Batch size per device. Usually kept between 1 and 4."},
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Batch size per device. Usually kept between 1 and 4."},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Steps of gradient accumulation. Kept between 4 and 8, it depends on the batch size."},
    )
    lr: Optional[float] = field(
        default=2e-5,
        metadata={"help": "The desired learning rate."},
    )
    lr_scheduler_type: Optional[str] = field(
        default='polynomial',
        metadata={"help": "The desired learning rate scheduler."},
    )
    lr_scheduler_kwargs_power: Optional[float] = field(
        default=1.0,
        metadata={"help": "The power of the polynomial learning rate."},
    )
    warmup_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Number of steps in which the learning rate in increased to the desired value."},
    )
    logging_strategy: Optional[str] = field(
        default='steps',
        metadata={"help": "The logging strategy. Use 'epoch' or 'steps'."},
    )
    logging_steps: Optional[int] = field(
        default=100,
        metadata={"help": "The logging steps. Only usable if logging_strategy is 'steps'."},
    )
    evaluation_strategy: Optional[str] = field(
        default='epoch',
        metadata={"help": "The evaluation strategy. Use 'epoch' or 'steps'."},
    )
    eval_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of steps between evaluations."},
    )
    save_strategy: Optional[str] = field(
        default='steps',
        metadata={"help": "The saving strategy. Use 'epoch' or 'steps'."},
    )

    output_dir: Optional[str] = field(
        default='./output',
        metadata={"help": "The output directory."},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "The seed for initializing training (reproducibility)."},
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


def register_wandb_project(
    model_args
) -> None:
    """
    Registers the Weights and Biases (wandb) project.

    Args:
        model_args (object): An object containing the model arguments.
                                Must contain wandb_project and wandb_run_name.

    Returns:
        None
    """
    wandb.init(project=model_args.wandb_project, name=model_args.wandb_run_name)


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
    if model_args.wandb_project is not None:
        register_wandb_project(model_args)
    main(model_args, data_args, training_args)
