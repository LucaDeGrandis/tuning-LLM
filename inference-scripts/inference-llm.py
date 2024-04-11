from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from typing import List, Any, Tuple, Union
from tqdm import tqdm

import os
import json
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference LLM")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained model or model identifier from Hugging Face model hub.")
    parser.add_argument("--peft_model_name_or_path", type=str, required=True, help="Path to the pre-trained peft model or model identifier from Hugging Face model hub.")
    parser.add_argument("--credentials_path", type=str, required=True, help="Path to the HuggingFace credentials.")
    parser.add_argument("--out_path", type=str, required=True, help="Where to save the generations.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data JSONL file.")
    parser.add_argument("--test_tasks_path", type=str, default=None, help="Path to the test data task IDs JSONL file.")
    parser.add_argument("--use_peft_lora", action="store_true", help="Use PEFT with LoRA.")
    parser.add_argument("--use_peft_pt", action="store_true", help="Use PEFT with Prompt Tuning.")
    parser.add_argument("--use_peft_mpt", action="store_true", help="Use PEFT with Multitask Prompt Tuning.")
    parser.add_argument("--custom_eos_token", type=str, default=None, help="A custom eos token to stop generation early (can drastically reduce inference time).")
    return parser.parse_args()


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


def write_json_file(
    filepath,
    input_dict,
    overwrite,
) -> None:
    if not overwrite:
        assert not os.path.exists(filepath)
    with open(filepath, 'w', encoding='utf8') as writer:
        json.dump(input_dict, writer, indent=4, ensure_ascii=False)


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


def create_datasets(
    args,
) -> Union[List[str], List[Tuple[str, int]]]:
    data_test = load_jsonl_file(args.test_data_path)

    if args.use_peft_mpt:
        assert args.test_tasks_path is not None
        data_test_ids = load_jsonl_file(args.test_tasks_path)
        assert len(data_test) == len(data_test_ids)
        data_test = [(prompt, id) for prompt, id in zip(data_test, data_test_ids)]

    print(f"A sample of test dataset: {data_test[0]}")

    return data_test


def create_and_prepare_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    if args.use_peft_lora or args.use_peft_pt or args.use_peft_mpt:
        model = PeftModel.from_pretrained(
            model,
            args.peft_model_name_or_path,
        )
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer


def create_generator(args, tokenizer):
    assert sum([args.use_peft_lora, args.use_peft_pt, args.use_peft_mpt]) == 1

    eos_token_id = tokenizer.eos_token_id
    if args.custom_eos_token is not None:
        eos_token_id = tokenizer.encode(args.custom_eos_token)

    if args.use_peft_lora:
        def generator(model, tokenizer, prompt, max_length):
            tok = tokenizer(prompt, return_tensors='pt').to('cuda')
            gen = model.generate(
                tok['input_ids'],
                max_length=max_length,
                temperature=0,
                eos_token_id=eos_token_id
            )[0]
            return tokenizer.decode(gen)
    elif args.use_peft_pt:
        def generator(model, tokenizer, prompt, max_length):
            tok = tokenizer(prompt, return_tensors='pt').to('cuda')
            gen = model.generate(
                input_ids=tok['input_ids'],
                attention_mask=tok['attention_mask'],
                max_length=max_length,
                temperature=0,
                eos_token_id=eos_token_id
            )[0]
            return tokenizer.decode(gen)
    elif args.use_peft_mpt:
        def generator(model, tokenizer, prompt, max_length):
            tok = tokenizer(prompt[0], return_tensors='pt')
            tok['task_ids'] = torch.tensor([prompt[1]])
            tok.to('cuda')
            gen = model.generate(
                input_ids=tok['input_ids'],
                attention_mask=tok['attention_mask'],
                task_ids=tok['task_ids'],
                max_length=max_length,
                temperature=0,
                eos_token_id=eos_token_id
            )[0]
            return tokenizer.decode(gen)
    else:
        def generator(model, tokenizer, prompt, max_length):
            tok = tokenizer(prompt, return_tensors='pt').to('cuda')
            gen = model.generate(
                tok['input_ids'],
                max_length=max_length,
                temperature=0,
                eos_token_id=eos_token_id
            )[0]
            return tokenizer.decode(gen)
    return generator


def main(args):
    assert args.credentials_path is not None, \
        "The credentials path must be provided. If you are loading a model that doesn't require credentials, use '--force_skip_credentials True'."
    hf_token = load_json_file(args.credentials_path)['private_token']
    os.environ['HF_TOKEN'] = hf_token

    # model
    model, tokenizer = create_and_prepare_model(args)

    # datasets
    test_dataset = create_datasets(args)

    # generator function
    generator = create_generator(args, tokenizer)

    # generate
    print('Saving results to:', args.out_path)
    generations = []
    for prompt in tqdm(test_dataset):
        generations.append(generator(model, tokenizer, prompt, max_length=2000))
    write_jsonl_file(
        args.out_path,
        generations,
        overwrite=True,
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
