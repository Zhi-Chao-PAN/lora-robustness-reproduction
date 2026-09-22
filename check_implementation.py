"""Substantive checks: published low-rank formula and real model freezing."""
import argparse
import gc
import json
from pathlib import Path
import torch
import run_experiment as experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(71)
    layer = experiment.lora.Linear(5, 4, r=2, lora_alpha=4, merge_weights=False, bias=True).double()
    layer.bias.requires_grad_(False)
    with torch.no_grad():
        layer.lora_B.normal_()
    x = torch.randn(3, 5, dtype=torch.float64)
    expected = torch.nn.functional.linear(x, layer.weight, layer.bias) + 2 * (x @ layer.lora_A.T @ layer.lora_B.T)
    error = float((layer(x) - expected).abs().max().detach())
    assert error < 1e-12
    fixed = layer.weight.detach().clone()
    layer(x).square().mean().backward()
    assert layer.weight.grad is None and layer.bias.grad is None
    assert layer.lora_A.grad.abs().sum() > 0 and layer.lora_B.grad.abs().sum() > 0
    optimizer = torch.optim.SGD([layer.lora_A, layer.lora_B], lr=0.01)
    optimizer.step()
    assert torch.equal(fixed, layer.weight)
    full, head_hash = experiment.make_model(args.asset_dir, 'full', 42)
    full.eval()
    tokens = torch.tensor([[0, 31414, 232, 2, 1, 1]])
    attention_mask = tokens.ne(1).long()
    with torch.no_grad():
        reference = full(input_ids=tokens, attention_mask=attention_mask).logits.clone()
    del full
    gc.collect()
    low_rank, lora_head_hash = experiment.make_model(args.asset_dir, 'lora_r8', 42)
    low_rank.eval()
    with torch.no_grad():
        adapted = low_rank(input_ids=tokens, attention_mask=attention_mask).logits
    assert head_hash == lora_head_hash
    assert torch.equal(reference, adapted), 'Zero LoRA update must preserve initial model output'
    trainable = [(name, p) for name, p in low_rank.named_parameters() if p.requires_grad]
    assert all(name.startswith('classifier.') or '.lora_A' in name or '.lora_B' in name for name, _ in trainable)
    assert sum(p.numel() for _, p in trainable) == 887042
    assert sum(1 for name, _ in trainable if '.lora_A' in name) == 24
    report = {
        'status': 'PASS', 'published_formula_max_abs_error': error,
        'base_parameters_receive_no_gradient': True, 'both_low_rank_factors_receive_gradient': True,
        'base_weights_unchanged_after_step': True, 'zero_adapter_real_model_logits_identical': True,
        'same_seed_classifier_initialization_identical': True,
        'query_value_layers_adapted': 24, 'lora_r8_trainable_parameters': 887042,
    }
    experiment.dump(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
