from core import SemanticAlignmentEmbeddingInitializer
from transformers import AutoModelForCausalLM, AutoTokenizer

source = "mii-community/zefiro-7b-base-ITA"
target = ""

source_model = AutoModelForCausalLM.from_pretrained(source)
helper_model = AutoModelForCausalLM.from_pretrained(target)
source_tokenizer = AutoTokenizer.from_pretrained(source)
target_tokenizer = AutoTokenizer.from_pretrained(target)

semanticAlignment = SemanticAlignmentEmbeddingInitializer(
    source_model,
    source_tokenizer,
    helper_model,
    target_tokenizer,
    False,
    42,
    False
    )


source_model_adapted = semanticAlignment()

source_model_adapted.push_to_hub("sapienzanlp/zefiro-7b_minestral_adapted_linear_5k_anchors_intersection_clean", private=True)
target_tokenizer.push_to_hub("sapienzanlp/zefiro-7b_minestral_adapted_linear_5k_anchors_intersection_clean", private=True)