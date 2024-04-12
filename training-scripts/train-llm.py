from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed
)
from peft import (
    get_peft_model,
    LoraConfig,
    PromptTuningConfig,
    MultitaskPromptTuningConfig,
    MultitaskPromptTuningInit,
)
from trl import SFTTrainer

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any

import torch
import json
import wandb
import os


@dataclass
class ModelArguments:
    """
    Arguments for the LLM model.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )

    ##################
    # Lora arguments #
    ##################
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "LoRA alpha parameter."},
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "LoRA dropout parameter."},
    )
    lora_r: Optional[int] = field(
        default=64,
        metadata={"help": "LoRA r parameter."},
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    save_to: Optional[str] = field(
        default='',
        metadata={"help": "Where to save the model."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Uses reentrant checkpointing implementation."},
    )

    ###########################
    # Prompt Tuning arguments #
    ###########################
    use_peft_pt: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT PT for training."},
    )
    pt_virtual_tokens: Optional[int] = field(
        default=20,
        metadata={
            "help": "The number of virtual tokens to prepend to the input."
        },
    )
    prompt_tuning_init_text: Optional[str] = field(
        default='',
        metadata={
            "help": "The path to the prompt tuning state dict used for initialization."
        },
    )
    pt_init_state_dict: Optional[str] = field(
        default=None,
        metadata={"help": "The state dict from which to train the target mpt."},
    )

    #####################################
    # Multitask Prompt Tuning arguments #
    #####################################
    # Must use some arguments from Prompt Tuning too:
    #   - pt_virtual_tokens
    #   - prompt_tuning_init_text
    use_peft_mpt: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT Multitask PT for training."},
    )
    pt_num_ranks: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of ranks for the low rank MPT matrices."
        },
    )
    pt_num_tasks: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of tasks for MPT."
        },
    )
    training_data_ids: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the task ids for the training dataset."
        },
    )
    dev_data_ids: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the task ids for the dev dataset."
        },
    )

    ###################
    # Wandb arguments #
    ###################
    wandb_project: Optional[str] = field(
        default='',
        metadata={"help": "Wandb project."},
    )
    wandb_run_name: Optional[str] = field(
        default='',
        metadata={"help": "Wandb run name."},
    )


@dataclass
class DataTrainingArguments:
    """
    DataTrainingArguments is a data class that holds the arguments for training a language model.
    """
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."}
    )
    max_seq_length: Optional[int] = field(default=512)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, appends `eos_token_id` at the end of each sample being packed."
        },
    )
    training_data_path: str = field(
        default='',
        metadata={
            "help": "The path to the jsonl traning dataset. The dataset must be a list of strings, each representing a complete example of prompt and model output."
        },
    )
    dev_data_path: str = field(
        default='',
        metadata={
            "help": "The path to the jsonl dev dataset. The dataset must be a list of strings, each representing a complete example of prompt and model output."
        },
    )
    train_shuffle_seed: int = field(
        default=42,
        metadata={
            "help": "The seed for shuffling the training dataset."
        },
    )
    credentials_path: str = field(
        default=None,
        metadata={
            "help": "The path to the credentials file. It must point to a json file with a 'private_token' key containg the HuggingFace private token."
        },
    )
    force_skip_credentials: bool = field(
        default=False,
        metadata={
            "help": "Whether to force skip loading the credential file. Set to True if the model you are loading is not protected by credentials."
        },
    )


def load_json_file(filepath):
    """
    Loads a JSON file and returns its contents as a Python dictionary.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a Python dictionary.
    """
    with open(filepath, 'r', encoding='utf8') as reader:
        json_data = json.load(reader)
    return json_data


def load_jsonl_file(
    filepath: str,
) -> List[Any]:
    """
    Loads a JSONL file and returns its contents as a list of dictionaries.

    Args:
        filepath (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries representing the contents of the JSONL file.
                If the jsonl was saved from a list, the function returns the list.
    """
    data = []
    with open(filepath, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data


def create_datasets(
    data_args, model_args,
) -> Tuple[Dataset, Dataset]:
    """
    Create train and validation datasets.

    Args:
        data_args (object): An object containing the data arguments.

    Returns:
        tuple: A tuple containing the train and validation datasets.
    """
    data_train_raw = load_jsonl_file(data_args.training_data_path)
    data_dev_raw = load_jsonl_file(data_args.dev_data_path)

    if model_args.use_peft_mpt:
        assert model_args.training_data_ids is not None
        assert model_args.dev_data_ids is not None
        data_train_ids = load_jsonl_file(model_args.training_data_ids)
        data_dev_ids = load_jsonl_file(model_args.dev_data_ids)
        assert len(data_train_ids) == len(data_train_raw)
        assert len(data_dev_ids) == len(data_dev_raw)
        train_data = Dataset.from_dict({'prompt': data_train_raw, 'task_ids': data_train_ids})
        valid_data = Dataset.from_dict({'prompt': data_dev_raw, 'task_ids': data_dev_ids})
    else:
        train_data = Dataset.from_dict({'prompt': data_train_raw})
        valid_data = Dataset.from_dict({'prompt': data_dev_raw})

    train_data = train_data.shuffle(seed=data_args.train_shuffle_seed)
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data


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
        # Check that only one peft model was selected
        assert sum([model_args.use_peft_lora, model_args.use_peft_pt, model_args.use_peft_mpt]) == 1, \
            "Only one PEFT model should be selected."

    def __call__(self):
        load_in_8bit = self.model_args.use_8bit_quantization
        torch_dtype = torch.float32

        bnb_config = None
        if load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                quantization_config=bnb_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                attn_implementation="flash_attention_2" if self.model_args.use_flash_attn else "eager",
                torch_dtype=torch_dtype,
            )

        model.config.use_cache = not self.training_args.gradient_checkpointing

        peft_config = self.create_config()

        model = get_peft_model(model, peft_config)

        return model

    def create_config(self):
        """Create the configuration for the PEFT model.

        Returns:
            object: The configuration object for the PEFT model.
        """
        peft_config = None

        if self.model_args.use_peft_lora:
            peft_config = LoraConfig(
                lora_alpha=self.model_args.lora_alpha,
                lora_dropout=self.model_args.lora_dropout,
                r=self.model_args.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.model_args.lora_target_modules.split(",")
                if self.model_args.lora_target_modules != "all-linear"
                else self.model_args.lora_target_modules,
            )

        if self.model_args.use_peft_pt:
            peft_config = PromptTuningConfig(
                num_virtual_tokens=self.model_args.pt_virtual_tokens,
                tokenizer_name_or_path=self.model_args.model_name_or_path,
                task_type="CAUSAL_LM",
                prompt_tuning_init='TEXT',
                prompt_tuning_init_text=self.model_args.prompt_tuning_init_text,
            )

        if self.model_args.use_peft_mpt:
            if self.model_args.pt_init_state_dict is None:
                peft_config = MultitaskPromptTuningConfig(
                    tokenizer_name_or_path=self.model_args.model_name_or_path,
                    num_tasks=self.model_args.pt_num_tasks,
                    num_ranks=self.model_args.pt_num_ranks,
                    task_type="CAUSAL_LM",
                    prompt_tuning_init='TEXT',
                    num_virtual_tokens=self.model_args.pt_virtual_tokens,
                    num_transformer_submodules=1,
                    prompt_tuning_init_text=self.model_args.prompt_tuning_init_text,
                )
            else:
                peft_config = MultitaskPromptTuningConfig(
                    tokenizer_name_or_path=self.model_args.model_name_or_path,
                    num_tasks=1,
                    num_ranks=self.model_args.pt_num_ranks,
                    task_type="CAUSAL_LM",
                    prompt_tuning_init=MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS,
                    prompt_tuning_init_state_dict_path=self.model_args.pt_init_state_dict,
                    num_virtual_tokens=self.model_args.pt_virtual_tokens,
                    num_transformer_submodules=1,
                )

        return peft_config

    def create_tokenizer(self):
        """Create the tokenizer.

        Returns:
            object: The tokenizer object.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer


def create_and_prepare_model(model_args, data_args, training_args):
    """
    Creates and prepares the model initializer for training.

    Args:
        model_args (object): The arguments for model initialization.
        data_args (object): The arguments for data preparation.
        training_args (object): The arguments for training.

    Returns:
        tuple: A tuple containing the model initialization object and the tokenizer.
    """
    model_init = ModelInit(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    tokenizer = model_init.create_tokenizer()

    return model_init, tokenizer


def main(model_args, data_args, training_args):
    assert data_args.credentials_path is not None, \
        "The credentials path must be provided. If you are loading a model that doesn't require credentials, use '--force_skip_credentials True'."
    if not data_args.force_skip_credentials:
        hf_token = load_json_file(data_args.credentials_path)['private_token']
        os.environ['HF_TOKEN'] = hf_token

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    model_init, tokenizer = create_and_prepare_model(
        model_args, data_args, training_args
    )

    # gradient ckpt
    training_args.gradient_checkpointing_kwargs = {
        "use_reentrant": model_args.use_reentrant
    }

    # datasets
    train_dataset, eval_dataset = create_datasets(
        data_args, model_args,
    )

    # trainer
    trainer = SFTTrainer(
        model_init=model_init,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=data_args.packing,
        dataset_kwargs={
            "append_concat_token": data_args.append_concat_token,
        },
        dataset_text_field="prompt",
        max_seq_length=data_args.max_seq_length,
    )
    trainer.accelerator.print(f"{trainer.model}")
    if model_args.use_peft_pt:
        trainer.model.print_trainable_parameters()

    trainer.train()

    # saving final model
    trainer.save_model(model_args.save_to)


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.report_to == ['wandb']:
        register_wandb_project(model_args)
    main(model_args, data_args, training_args)
