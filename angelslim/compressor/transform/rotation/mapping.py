# adopted from llm-compressor/src/llmcompressor/modifiers/transform/spinquant/mappings.py


__all__ = ["linear_mapping", "norm_mapping"]


linear_mapping = dict(
    embedding="embed_tokens",
    attn="self_attn",
    attn_q="q_proj",
    attn_k="k_proj",
    attn_v="v_proj",
    attn_o="o_proj",
    mlp_in=["up_proj", "gate_proj"],
    mlp_out=["down_proj"],
    lm_head="lm_head",
)

# Each entry is (to_linear_list, to_norm),
# matching get_rotation_mapping_layers norm_mapping format.
# Longest-prefix matching is used to support MoE experts.
norm_mapping = [
    (["q_proj", "k_proj", "v_proj"], "input_layernorm"),
    (["up_proj", "gate_proj"], "post_attention_layernorm"),
    (["lm_head"], "norm"),
]
