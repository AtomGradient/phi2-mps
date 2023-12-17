import argparse
import mlx.core as mx
from . import phi2

parser = argparse.ArgumentParser(description="Phi-2 inference script")

# model path should be type of [npz]
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="where the phi2 model store in local",
)
parser.add_argument(
    "--prompt",
    help="The message to be processed by the model",
    default="Write a detailed analogy between mathematics and a lighthouse.",
)
parser.add_argument(
    "--max_tokens",
    "-m",
    type=int,
    default=100,
    help="Maximum number of tokens to generate",
)
parser.add_argument(
    "--temp",
    help="The sampling temperature.",
    type=float,
    default=0.0,
)
parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

def main():
    args = parser.parse_args()
    mx.random.seed(args.seed)
    model, tokenizer = phi2.load_model(args.model)
    prompt = tokenizer(
        args.prompt,
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"]

    prompt = mx.array(prompt)

    print("[INFO] Generating with Phi-2...", flush=True)
    print(args.prompt, end="", flush=True)

    tokens = []
    for token, _ in zip(phi2.generate(prompt, model, args.temp), range(args.max_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            mx.eval(tokens)
            eos_index = next(
                (i for i, t in enumerate(tokens) if t.item() == tokenizer.eos_token_id),
                None,
            )

            if eos_index is not None:
                tokens = tokens[:eos_index]

            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []
            if eos_index is not None:
                break

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)
