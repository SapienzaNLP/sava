from pathlib import Path
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import (CLPEmbeddingInitializer, 
                  SemanticAlignmentEmbeddingInitializer,
                  AverageEmbeddingInitializer,
                  CLPRandomEmbeddingInitializer)


def manage_tokenizer_union(source_tokenizer, target_tokenizer, output_dir):
    """
    Adjust tokenizer fields like new tokens and merges
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdirname:
        path_tmp_dir = Path(tmpdirname)
        path_source_tokenzizer = path_tmp_dir / "source_tok"
        path_target_tokenzizer = path_tmp_dir / "target_tok"

        source_tokenizer.save_pretrained(path_source_tokenzizer)
        target_tokenizer.save_pretrained(path_target_tokenzizer)

        target_token_to_idx = {t: i for t, i in target_tokenizer.get_vocab().items()}
        source_token_to_idx = {t: i for t, i in source_tokenizer.get_vocab().items()}

        target_token_to_new_idx = dict()

        target_vocabulary_out = source_token_to_idx.copy()
        new_token_count = 0
        for token, target_idx in target_token_to_idx.items():
            target_token_to_new_idx[token] = target_idx

            if token not in source_token_to_idx:
                target_vocabulary_out[token] = len(source_token_to_idx) + new_token_count

                new_token_count += 1



        merges = []

        # first add the target merges e.g italian merges
        with open(path_target_tokenzizer / "tokenizer.json") as fr:
            data = json.loads(fr.read())
            merges.extend(data["model"]["merges"])

        # then add the english original ones
        with open(path_source_tokenzizer / "tokenizer.json") as fr:
            original = json.loads(fr.read())
            set_merges = set(merges)
            intersection = set_merges.intersection(set(original["model"]["merges"]))
            for m in original["model"]["merges"]:
                if m not in intersection:
                    merges.append(m)

        with open( Path(output_dir) / "tokenizer.json") as fr:
            data = json.loads(fr.read())
        data["model"]["merges"] = list(merges)
        data["model"]["vocab"] = target_vocabulary_out
        # copy the added tokens from the original tokenizer
        data["added_tokens"] = original["added_tokens"]
        with open(Path(output_dir) / "tokenizer.json", "w") as fw:
            fw.write(json.dumps(data, indent=4))
        shutil.copyfile(path_source_tokenzizer / "tokenizer_config.json", Path(output_dir) / "tokenizer_config.json")


def main(args):
    if args.initialization_method == "random":
        target_tokenizer = AutoTokenizer.from_pretrained(
            args.target_tokenizer_name_or_path,
            cache_dir=args.cache_dir
        )
        initializer = RandomEmbeddingInitializer(
            source_model=AutoModelForCausalLM.from_pretrained(
                args.source_model_name_or_path,
                torch_dtype=torch.float16,            
                cache_dir=args.cache_dir
            ),
            source_tokenizer=AutoTokenizer.from_pretrained(
                args.source_tokenizer_name_or_path,
                cache_dir=args.cache_dir
            ),
            target_tokenizer=target_tokenizer,
            seed=args.seed,
            tie_weights=args.tie_weights,
        )
        target_model = initializer()
        target_model.save_pretrained(args.output_dir)
        target_tokenizer.save_pretrained(args.output_dir)
        
    elif args.initialization_method == "clp":
        target_tokenizer = AutoTokenizer.from_pretrained(
            args.target_tokenizer_name_or_path,
            cache_dir=args.cache_dir
        )
        source_tokenizer=AutoTokenizer.from_pretrained(
                args.source_tokenizer_name_or_path,
                cache_dir=args.cache_dir
            )
        initializer = CLPEmbeddingInitializer(
            source_model=AutoModelForCausalLM.from_pretrained(
                args.source_model_name_or_path,
                torch_dtype=torch.float16,
                cache_dir=args.cache_dir
            ),
            helper_model=AutoModelForCausalLM.from_pretrained(
                args.helper_model_name_or_path,
                torch_dtype=torch.float16,
                cache_dir=args.cache_dir
            ),
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            target_vocabulary_strategy=args.target_vocabulary_strategy,
            copy_special_tokens=args.copy_special_tokens,
            tie_weights=args.tie_weights,
            seed=args.seed
        )
        target_model = initializer()
        target_model.save_pretrained(args.output_dir)

        if args.target_vocabulary_strategy == "union":
            source_tokenizer.save_pretrained(args.output_dir)
            manage_tokenizer_union(target_tokenizer=AutoTokenizer.from_pretrained(args.target_tokenizer_name_or_path,
                                                                                  cache_dir=args.cache_dir),
                                   source_tokenizer=AutoTokenizer.from_pretrained(args.source_tokenizer_name_or_path,
                                                                                  cache_dir=args.cache_dir),
                                   output_dir=args.output_dir)
        else:
            target_tokenizer.save_pretrained(args.output_dir)
    
    elif args.initialization_method == "average_naive":
        target_tokenizer = AutoTokenizer.from_pretrained(
            args.target_tokenizer_name_or_path,
            cache_dir=args.cache_dir
        )
        initializer = AverageEmbeddingInitializer(
            target_model=AutoModelForCausalLM.from_pretrained(
                args.source_model_name_or_path,
                torch_dtype=torch.float16,
                cache_dir=args.cache_dir
            ),
            source_tokenizer=AutoTokenizer.from_pretrained(
                args.source_tokenizer_name_or_path,
                cache_dir=args.cache_dir
            ),
            target_tokenizer=target_tokenizer,
            copy_special_tokens=args.copy_special_tokens,
            seed=args.seed,
            tie_weights=args.tie_weights
        )
        target_model = initializer()
        target_model.save_pretrained(args.output_dir)
        target_tokenizer.save_pretrained(args.output_dir)

    elif args.initialization_method == "semantic_alignment":
        target_tokenizer=AutoTokenizer.from_pretrained(
                args.target_tokenizer_name_or_path,
                cache_dir=args.cache_dir
            )
        source_tokenizer = AutoTokenizer.from_pretrained(
                args.source_tokenizer_name_or_path,
                cache_dir=args.cache_dir,
            )
        initializer = SemanticAlignmentEmbeddingInitializer(
            source_model=AutoModelForCausalLM.from_pretrained(
                args.source_model_name_or_path,
                torch_dtype=torch.float16,
                cache_dir=args.cache_dir,
            ),
            source_tokenizer=source_tokenizer,
            helper_model=AutoModelForCausalLM.from_pretrained(
                args.helper_model_name_or_path,
                torch_dtype=torch.float16,
                cache_dir=args.cache_dir,
            ),
            target_tokenizer=target_tokenizer,
            seed=args.seed,
            tie_weights=args.tie_weights,
            aligner=args.aligner,
            num_anchors=args.num_anchors,
            substitute_intersection=args.substitute_intersection,
            target_vocabulary_strategy=args.target_vocabulary_strategy,
            anchor_selection=args.anchor_selection
        )
        target_model = initializer()
        target_model.save_pretrained(args.output_dir)
        if args.target_vocabulary_strategy == "union":
            source_tokenizer.save_pretrained(args.output_dir)
            manage_tokenizer_union(target_tokenizer=AutoTokenizer.from_pretrained(args.target_tokenizer_name_or_path,
                                                                                  cache_dir=args.cache_dir),
                                   source_tokenizer=AutoTokenizer.from_pretrained(args.source_tokenizer_name_or_path,
                                                                                  cache_dir=args.cache_dir),
                                   output_dir=args.output_dir)
        else:
            target_tokenizer.save_pretrained(args.output_dir)
    elif args.initialization_method == "clp_random":
        target_tokenizer=AutoTokenizer.from_pretrained(
                args.target_tokenizer_name_or_path,
                cache_dir=args.cache_dir
            )
        source_tokenizer = AutoTokenizer.from_pretrained(
                args.source_tokenizer_name_or_path,
                cache_dir=args.cache_dir,
            )
        initializer = CLPRandomEmbeddingInitializer(
            source_model=AutoModelForCausalLM.from_pretrained(
                args.source_model_name_or_path,
                torch_dtype=torch.float16,
                cache_dir=args.cache_dir,
            ),
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            seed=args.seed,
            tie_weights=args.tie_weights,
            target_vocabulary_strategy=args.target_vocabulary_strategy
        )
        target_model = initializer()
        target_model.save_pretrained(args.output_dir)
        if args.target_vocabulary_strategy == "union":

            source_tokenizer.save_pretrained(args.output_dir)
            manage_tokenizer_union(target_tokenizer=AutoTokenizer.from_pretrained(args.target_tokenizer_name_or_path,
                                                                                  cache_dir=args.cache_dir),
                                   source_tokenizer=AutoTokenizer.from_pretrained(args.source_tokenizer_name_or_path,
                                                                                  cache_dir=args.cache_dir),
                                   output_dir=args.output_dir)
        else:
            target_tokenizer.save_pretrained(args.output_dir)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialization_method",
        type=str,
        choices=["clp", "clp_random", "average_naive", "semantic_alignment"],
        required=True,
        help="The embedding initialization method to use."
    )
    parser.add_argument(
        "--source_model_name_or_path",
        type=str, 
        required=True,
        help="The source model to initialize the target model with."
    )
    parser.add_argument(
        "--source_tokenizer_name_or_path", 
        type=str, 
        required=True,
        help="The source tokenizer to initialize the target tokenizer with."
    )
    parser.add_argument(
        "--target_tokenizer_name_or_path",
        type=str,
        required=True,
        default=None,
        help="The helper model tokenizer."
    )
    parser.add_argument(
        "--helper_model_name_or_path", 
        type=str,
        required=False,
        default=None,
        help="[clp, semantic] The helper model to help initialize a terget model."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The cache directory to save the pretrained models."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="The random seed."
    )
    parser.add_argument(
        "--copy_special_tokens", 
        action="store_true",
        help="[clp, clp_plus] Whether to copy the special tokens' embeddings from the source model to the target model."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="The output directory to save the target model and tokenizer."
    )
    parser.add_argument(
        "--tie_weights",
        action="store_true",
        help="Whether to tie the weights of the target model."
    )
    parser.add_argument(
        "--aligner",
        type=str,
        required=False,
        default=None,
        help="[semantic] Aligner name."
    )
    parser.add_argument(
        "--substitute_intersection",
        action="store_true",
        help="[semantic] If we want to substitute the vocabulary intersection with the ."
    )
    parser.add_argument(
        "--target_vocabulary_strategy",
        type=str,
        help="The strategy to use for vocabulary ['union', 'substitute']",
        default="substitute"
    )
    parser.add_argument(
        "--anchor_selection",
        type=str,
        help="[semantic] The strategy to select anchors ['random', 'full']",
        default="random"
    )
    parser.add_argument(
        "--num_anchors",
        type=int,
        required=False,
        default=None,
        help="[semantic] number of anchors."
    )

    args = parser.parse_args()
    main(args)