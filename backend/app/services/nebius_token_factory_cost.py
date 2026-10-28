"""Estimate Token Factory token cost from a downloaded pricing catalog.

Runtime mode is approximate: generation time × output tokens/second.
For completed requests, pass actual token counts from API usage instead.

Uses Python's standard library only.
"""

import argparse
import json
from decimal import Decimal
from pathlib import Path

MILLION = Decimal("1000000")


def nonnegative(value, name):
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError(f"{name} must be a finite, nonnegative number.")
    return number


def find_pricing(catalog, model_id, region=None):
    matches = []

    for model in catalog:
        for flavor in model.get("flavors", []):
            if flavor.get("model_id") != model_id:
                continue

            regions = [
                item["name"]
                for item in flavor.get("regions", [])
                if "name" in item
            ]

            if region is None or region in regions:
                matches.append(flavor)

    if not matches:
        raise ValueError(
            f"No pricing found for {model_id!r}, region={region!r}."
        )

    if len(matches) > 1:
        raise ValueError(
            "Multiple pricing entries match. Specify a region or "
            "select a unique catalog entry."
        )

    return matches[0]


def estimate_cost(
    pricing,
    *,
    input_tokens,
    cached_input_tokens=0,
    output_tokens=None,
    generation_seconds=None,
    output_tokens_per_second=None,
):
    """input_tokens includes cached_input_tokens.

    Token counts and generation time refer to one request.
    """

    for name, value in (
        ("input_tokens", input_tokens),
        ("cached_input_tokens", cached_input_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")

    if cached_input_tokens > input_tokens:
        raise ValueError("Cached input cannot exceed total input tokens.")

    if output_tokens is not None:
        if generation_seconds is not None or output_tokens_per_second is not None:
            raise ValueError(
                "Provide output_tokens OR generation time and throughput."
            )
        if (
            isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or output_tokens < 0
        ):
            raise ValueError("output_tokens must be a nonnegative integer.")

        output_count = Decimal(output_tokens)
        method = "provided_token_counts"

    else:
        if generation_seconds is None or output_tokens_per_second is None:
            raise ValueError(
                "Runtime mode requires generation_seconds and "
                "output_tokens_per_second."
            )

        seconds = nonnegative(generation_seconds, "generation_seconds")
        throughput = nonnegative(
            output_tokens_per_second, "output_tokens_per_second"
        )
        if throughput == 0:
            raise ValueError("Output throughput must be greater than zero.")

        output_count = seconds * throughput
        method = "estimated_from_generation_time"

    input_rate = nonnegative(
        pricing["input_price_per_million_tokens"], "input price"
    )
    output_rate = nonnegative(
        pricing["output_price_per_million_tokens"], "output price"
    )

    cached_rate = None
    if cached_input_tokens:
        value = pricing.get("input_cached_price_per_million_tokens")
        if value is None:
            raise ValueError("Catalog has no cached-input price.")
        cached_rate = nonnegative(value, "cached input price")

    uncached_count = input_tokens - cached_input_tokens

    input_cost = Decimal(uncached_count) * input_rate / MILLION
    cached_cost = (
        Decimal(cached_input_tokens) * cached_rate / MILLION
        if cached_input_tokens
        else Decimal(0)
    )
    output_cost = output_count * output_rate / MILLION

    return {
        "model_id": pricing["model_id"],
        "currency": "USD",
        "method": method,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": float(output_count),
        "input_cost_usd": str(input_cost),
        "cached_input_cost_usd": str(cached_cost),
        "output_cost_usd": str(output_cost),
        "total_cost_usd": str(input_cost + cached_cost + output_cost),
        "assumptions": (
            [
                "Runtime is output generation time, not total request latency.",
                "Output throughput is an assumed average for this request.",
                "Generated tokens represent all billable output tokens.",
            ]
            if method == "estimated_from_generation_time"
            else []
        ),
        "exclusions": [
            "Taxes and account-specific discounts.",
            "Dedicated endpoint GPU-hour charges.",
            "Tool calls or other separately billed services.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--region")
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--cached-input-tokens", type=int, default=0)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--generation-seconds", type=float)
    parser.add_argument("--output-tps", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        catalog = json.loads(args.pricing.read_text(encoding="utf-8"))
        pricing = find_pricing(catalog, args.model, args.region)

        result = estimate_cost(
            pricing,
            input_tokens=args.input_tokens,
            cached_input_tokens=args.cached_input_tokens,
            output_tokens=args.output_tokens,
            generation_seconds=args.generation_seconds,
            output_tokens_per_second=args.output_tps,
        )

        text = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)

    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(1, f"Cost estimation failed: {exc}\n")


if __name__ == "__main__":
    main()