"""Deterministic LLM sizing arithmetic. Input JSON on stdin; output JSON on stdout."""
import json
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import sys


def number(obj, key, *, whole=False, zero=False):
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{key} must be a number")
    value = Decimal(str(value))
    if not value.is_finite() or value < 0 or (not zero and value == 0):
        raise ValueError(f"{key} must be finite and {'nonnegative' if zero else 'positive'}")
    if whole and value != value.to_integral_value():
        raise ValueError(f"{key} must be an integer")
    return value


def ceiling(value):
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def peak(obj):
    target = number(obj, "target_pflops")
    per_gpu = number(obj, "per_gpu_tflops")
    per_node = int(number(obj, "gpus_per_node", whole=True))
    cards = ceiling(target * 1000 / per_gpu)
    nodes = (cards + per_node - 1) // per_node
    return {"minimum_cards": cards, "full_nodes": nodes, "full_node_cards": nodes * per_node,
            "full_node_peak_pflops": float(nodes * per_node * per_gpu / 1000)}


def memory(obj):
    weights = number(obj, "parameters", whole=True) * number(obj, "weight_bits", whole=True) / 8
    if "kv_bytes_per_token" in obj:
        if any(k in obj for k in ("layers", "kv_heads", "head_dim", "kv_bits")):
            raise ValueError("choose explicit KV bytes/token or standard-attention inputs, not both")
        kv_per_token = number(obj, "kv_bytes_per_token")
    else:
        kv_per_token = 2 * number(obj, "layers", whole=True) * number(obj, "kv_heads", whole=True) * (
            number(obj, "head_dim", whole=True) * number(obj, "kv_bits", whole=True) / 8)
    kv = kv_per_token * number(obj, "live_tokens", whole=True) * number(obj, "sequences", whole=True)
    candidates = obj["tp_candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("tp_candidates must be a nonempty list of positive integers")
    tp_widths = sorted({int(number({"TP": t}, "TP", whole=True)) for t in candidates})
    gib = 1024 ** 3
    reserve = number(obj, "reserve_fraction", zero=True)
    if reserve >= 1:
        raise ValueError("reserve_fraction must be in [0, 1)")
    usable = number(obj, "gpu_vram_gib") * (1 - reserve)
    mode = obj.get("kv_sharding", "replicated")
    if mode not in ("replicated", "ideal"):
        raise ValueError("kv_sharding must be replicated or ideal")
    loads = [(t, (weights / t + (kv / t if mode == "ideal" else kv)) / gib) for t in tp_widths]
    return {"weight_gib": float(weights / gib), "kv_gib_per_replica": float(kv / gib),
            "usable_gib_per_card": float(usable), "kv_sharding": mode,
            "estimated_gib_per_card_by_tp": {str(t): float(load) for t, load in loads},
            "minimum_tested_tp": next((t for t, load in loads if load <= usable), None)}


def service(obj):
    capacity = number(obj, "qualified_requests_per_second_per_replica")
    factor = number({"capacity_fraction": obj.get("capacity_fraction", 1)}, "capacity_fraction")
    if factor > 1:
        raise ValueError("capacity_fraction must be in (0, 1]")
    cards = int(number(obj, "cards_per_replica", whole=True))
    replicas = ceiling(number(obj, "peak_requests_per_second") / (capacity * factor))
    return {"replicas": replicas, "total_cards": replicas * cards}


def main():
    try:
        request = json.load(sys.stdin, parse_float=Decimal)
        modes = {"peak": peak, "memory": memory, "service": service}
        if not isinstance(request, dict) or not request or request.keys() - modes.keys():
            raise ValueError("provide one or more of peak, memory, and service")
        if not all(isinstance(obj, dict) for obj in request.values()):
            raise ValueError("each mode must contain a JSON object")
        result = {name: modes[name](obj) for name, obj in request.items()}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (AttributeError, KeyError, TypeError, ValueError, InvalidOperation, OverflowError) as exc:
        print(f"Invalid sizing input: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
