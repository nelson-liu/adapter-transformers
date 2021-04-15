import argparse
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelWithHeads, AutoTokenizer
from truncating_pipeline import TruncatingTextClassificationPipeline


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main(data_path, model, adapter, adapter_type, batch_size, output_path, cuda_device, padding, max_length):
    print(f"Using device: {cuda_device}")
    print(f"Loading model and tokenizer for {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelWithHeads.from_pretrained(model)
    adapter_name = model.load_adapter(adapter, config=adapter_type)
    model.set_active_adapters(adapter_name)

    sentences = []
    with open(data_path) as f:
        for line in f:
            example = json.loads(line)
            sentences.append(example["text"])
    print(f"Read {len(sentences)} sentences")

    pipeline = TruncatingTextClassificationPipeline(model=model, tokenizer=tokenizer, device=cuda_device)
    print(f"Processing with a batch size of {batch_size}")

    predictions = []
    for batch in tqdm(chunks(sentences, batch_size)):
        with torch.no_grad():
            model_output = pipeline(inputs=batch,
                                    max_length=max_length,
                                    padding="max_length" if padding else padding,
                                    truncation=True)
            predictions.extend(model_output)

    assert len(sentences) == len(predictions)

    print(f"Saving to {output_path}")
    with open(output_path, "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to sentiment examples, jsonl.",
    )
    parser.add_argument(
        "--model", type=str, help="HuggingFace model to use"
    )
    parser.add_argument(
        "--adapter", type=str, help="Adapter model to use"
    )
    parser.add_argument(
        "--adapter-type", type=str, help="Adapter type"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size to use")
    parser.add_argument("--output-path", type=str, help="Path to save output")
    parser.add_argument("--cuda-device", type=int, default=-1, help="CUDA device to run on")
    parser.add_argument("--pad-to-max-length",
                        action="store_true",
                        default=True,
                        help="Whether to pad all samples to `max_seq_length`.")
    parser.add_argument("--max-seq-length", type=int, required=True, help="Maximum sequence length to use")
    args = parser.parse_args()
    main(args.data_path, args.model, args.adapter, args.adapter_type, args.batch_size, args.output_path, args.cuda_device, args.pad_to_max_length, args.max_seq_length)
