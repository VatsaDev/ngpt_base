"""
Implement actual benchmarks code here, like MMLU, gs8mk, etc
"""

# basically extending sampler to work on benchmarks, need this on top of loss because we have to evaluate downstream tasks. 

# obscure bench links

# LAMBADA - good for lang modelling, https://paperswithcode.com/sota/language-modelling-on-lambada
# PIQA - https://huggingface.co/datasets/ybisk/piqa
# Hellaswag - https://huggingface.co/datasets/Rowan/hellaswag
# Winograde - https://paperswithcode.com/sota/common-sense-reasoning-on-winogrande
# Humaneval - https://github.com/openai/human-eval
# IFEval - https://huggingface.co/datasets/google/IFEval
# BBH  - https://huggingface.co/datasets/lukaemon/bbh
# MATH - https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark
# GPQA - highly unlikely our model cracks this, but https://arxiv.org/abs/2311.12022
# MuSR - https://arxiv.org/abs/2310.16049

Benchmark_list = ["MMLU", "GS8MK", "LAMBADA", "PIQA", "Hellaswag", "Winograde", "Humaneval", "IFEval", "BBH", "Math", "GPQA", "MuSR"]
