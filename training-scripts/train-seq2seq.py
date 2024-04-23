from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
    HfArgumentParser
)

from dataclasses import dataclass, field

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
    source_max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum length for the source documents."},
    )
    target_max_length: Optional[int] = field(
        default=200,
        metadata={"help": "The maximum length for the output summaries."},
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
    save_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Number of steps between model checkpoints."},
    )
    output_dir: Optional[str] = field(
        default='./output',
        metadata={"help": "The output directory."},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "The seed for initializing training (reproducibility)."},
    )
    training_seed: Optional[int] = field(
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


def process_line(line):
    return line.strip()


def tokenize(
    line: Dict[str, str],
    data_args,
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
    tok_input = tokenizer(line['input'], truncation=True, padding='max_length', max_length=data_args.source_max_length)
    tok_output = tokenizer(line['output'], truncation=True, padding='max_length', max_length=data_args.target_max_length)
    return {
        'input_ids': tok_input['input_ids'],
        'attention_mask': tok_input['attention_mask'],
        'labels': tok_output['input_ids'],
    }


def create_datasets(
    data_args, tokenizer,
):
    """Create train and dev datasets.

    Args:
        data_args (object): The data arguments.
            The data must be stored in jsonl files. Each line of the file must be a string.
            For each train dataset (train/valid) you need:
                - A source file with the input documents (one per line)
                - A target file with the output summaries (one per line)

    Returns:
        tuple: A tuple containing the train dataset and dev dataset.
    """
    # Train data
    train_source = list(map(process_line, load_txt_file(data_args.train_source)))
    train_target = list(map(process_line, load_txt_file(data_args.train_target)))
    train_dataset = Dataset.from_dict({'input': train_source, 'output': train_target})
    train_dataset = train_dataset.map(lambda data: tokenize(data, data_args, tokenizer), batched=True)
    train_dataset = train_dataset.remove_columns(['input', 'output'])

    # Dev data
    dev_source = list(map(process_line, load_txt_file(data_args.dev_source)))
    dev_target = list(map(process_line, load_txt_file(data_args.dev_target)))
    dev_dataset = Dataset.from_dict({'input': dev_source, 'output': dev_target})
    dev_dataset = dev_dataset.map(lambda data: tokenize(data, data_args, tokenizer), batched=True)
    dev_dataset = dev_dataset.remove_columns(['input', 'output'])

    return train_dataset, dev_dataset


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

    tokenizer = model_init.create_tokenizer()

    return model_init, tokenizer


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    model_init, tokenizer = create_and_prepare_model(
        model_args, data_args, training_args
    )

    # datasets
    train_dataset, eval_dataset = create_datasets(
        data_args, tokenizer
    )

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        overwrite_output_dir=training_args.overwrite_output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        evaluation_strategy=training_args.evaluation_strategy,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        warmup_steps=training_args.warmup_steps,
        learning_rate=training_args.lr,
        logging_strategy=training_args.logging_strategy,
        logging_steps=training_args.logging_steps,
        lr_scheduler_type=training_args.lr_scheduler_type,
        lr_scheduler_kwargs={'power': training_args.lr_scheduler_kwargs_power},
        output_dir=training_args.output_dir,
        seed=training_args.training_seed,
    )

    # Create the Trainer instance
    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.wandb_project is not None:
        register_wandb_project(model_args)
    main(model_args, data_args, training_args)
