import os
import glob
from pathlib import Path
import argparse
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from huggingface_hub import login
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_text_files(data_dir, split):
    """Read all .{split} files from the directory"""
    pattern = f"{data_dir}/{split}/*.{split}"
    files = glob.glob(pattern)
    
    if not files:
        logger.warning(f"No {split} files found in {data_dir}/{split}/")
        return []
    
    texts = []
    file_sources = []
    
    for file_path in sorted(files):
        logger.info(f"Reading {file_path}")
        filename = Path(file_path).stem  # abc.train -> abc
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        texts.append(line)
                        file_sources.append(f"{filename}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    logger.info(f"Loaded {len(texts)} texts from {len(files)} {split} files")
    return texts, file_sources

def create_dataset_dict(data_dir):
    """Create DatasetDict from train/dev/test structure"""
    dataset_dict = {}
    
    for split in ['train', 'dev', 'test']:
        split_dir = Path(data_dir) / split
        if split_dir.exists():
            texts, sources = read_text_files(data_dir, split)
            if texts:
                dataset_dict[split] = Dataset.from_dict({
                    'text': texts,
                    'source_file': sources
                })
                logger.info(f"{split}: {len(texts)} examples")
            else:
                logger.warning(f"No data found for {split} split")
    
    if not dataset_dict:
        raise ValueError(f"No data found in {data_dir}")
    
    return DatasetDict(dataset_dict)

def tokenize_dataset(dataset_dict, tokenizer_name, max_length=512):
    """Tokenize the dataset"""
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # GPT-2 specific setup
    if "gpt2" in tokenizer_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("GPT-2 tokenizer: Set pad_token = eos_token")
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        # Tokenize without padding (dynamic padding is better)
        result = tokenizer(
            examples['text'], 
            truncation=True, 
            max_length=max_length,
            padding=False,
            return_tensors=None  # Keep as lists for datasets
        )
        
        # Add sequence length for analysis
        result['length'] = [len(ids) for ids in result['input_ids']]
        return result
    
    logger.info("Tokenizing dataset...")
    tokenized_dict = dataset_dict.map(
        tokenize_function,
        batched=True,
        num_proc=4,  # Parallel processing
        desc="Tokenizing"
    )
    
    # Print some stats
    for split_name, split_data in tokenized_dict.items():
        lengths = split_data['length']
        avg_len = sum(lengths) / len(lengths)
        max_len = max(lengths)
        logger.info(f"{split_name}: avg_length={avg_len:.1f}, max_length={max_len}")
    
    return tokenized_dict, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Tokenize and push dataset to HF Hub")
    parser.add_argument("--data_dir", required=True, help="Directory with train/dev/test folders")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name (e.g., bert-base-uncased)")
    parser.add_argument("--dataset_name", required=True, help="HF dataset name (e.g., username/my-dataset)")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--local_cache", help="Local directory to save dataset before pushing")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    parser.add_argument("--dry_run", action="store_true", help="Don't push to hub, just process locally")
    
    args = parser.parse_args()
    
    # Check data directory structure
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Create dataset
    logger.info("Creating dataset from files...")
    dataset_dict = create_dataset_dict(args.data_dir)
    
    # Tokenize
    logger.info("Tokenizing dataset...")
    tokenized_dict, tokenizer = tokenize_dataset(dataset_dict, args.tokenizer, args.max_length)
    
    # Save locally if requested
    if args.local_cache:
        logger.info(f"Saving to local cache: {args.local_cache}")
        tokenized_dict.save_to_disk(args.local_cache)
    
    # Push to hub
    if not args.dry_run:
        logger.info("Logging into HuggingFace...")
        login()  # Will prompt for token if not already logged in
        
        logger.info(f"Pushing dataset to {args.dataset_name}")
        tokenized_dict.push_to_hub(
            args.dataset_name,
            private=args.private
        )
        
        # Also push tokenizer for convenience
        tokenizer_name = f"{args.dataset_name}-tokenizer"
        logger.info(f"Pushing tokenizer to {tokenizer_name}")
        tokenizer.push_to_hub(tokenizer_name, private=args.private)
        
        logger.info("✅ Dataset and tokenizer pushed successfully!")
        logger.info(f"Load with: dataset = load_dataset('{args.dataset_name}')")
        logger.info(f"Load tokenizer: tokenizer = AutoTokenizer.from_pretrained('{tokenizer_name}')")
    else:
        logger.info("Dry run completed - no data pushed to hub")

if __name__ == "__main__":
    main()


# Example usage with GPT-2:
"""
python scripts/tokenize_data.py \
    --data_dir ./data \
    --tokenizer gpt2 \
    --dataset_name bendemonium/babylm25_tokens \
    --max_length 1024
"""