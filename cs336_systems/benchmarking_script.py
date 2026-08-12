import torch
import timeit
import modal
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

app = modal.App("cs336-benchmarking")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch~=2.11.0", "einops>=0.8", "einx>=0.4", "jaxtyping>=0.3", "numpy>=2.4")
    .add_local_python_source("cs336_basics")
)

@app.function(image=image, gpu="B200", timeout=1800)
def benchmarking_script(d_model, d_ff, num_layers, num_heads, w, n):
    """
    A script that will initialize a basics Transformer model with the given
    hyperparameters, create a random batch of data, and time forward-only, forward-and-
    backward, and full training steps that include the optimizer step
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize model based on given hyperparameters
    model = BasicsTransformerLM(vocab_size=10000, context_length=512, 
                                d_model=d_model, num_layers=num_layers, 
                                num_heads=num_heads, d_ff=d_ff).to(device)

    # generate random batch of data
    batch_size = 4
    vocab_size = 10000
    context_length = 512
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    optimizer = AdamW(model.parameters()) 

    # forward only
    def forward_only():
        # warmup steps
        for _ in range(w):
            optimizer.zero_grad()
            model(x)

        # timed steps
        times = []
        torch.cuda.synchronize() if device == "cuda" else None
        for _ in range(n):
            optimizer.zero_grad()
            start_time = timeit.default_timer()
            model(x)
            torch.cuda.synchronize() if device == "cuda" else None
            times.append(timeit.default_timer() - start_time)

        times = torch.tensor(times)
        return times

    def forward_and_backward():
        # warmup steps
        for _ in range(w):
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()


        # timed steps
        times = []
        torch.cuda.synchronize() if device == "cuda" else None
        for _ in range(n):
            optimizer.zero_grad()
            start_time = timeit.default_timer()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.cuda.synchronize() if device == "cuda" else None
            times.append(timeit.default_timer() - start_time)

        times = torch.tensor(times)
        return times

    def full():
        # warmup steps
        for _ in range(w):
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

        # timed steps
        times = []
        torch.cuda.synchronize() if device == "cuda" else None
        for _ in range(n):
            optimizer.zero_grad()
            start_time = timeit.default_timer()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize() if device == "cuda" else None
            times.append(timeit.default_timer() - start_time)

        times = torch.tensor(times)
        return times

    fwd = forward_only()
    fwdbwd = forward_and_backward()
    fl = full()
    return {
        "forward": {"mean": fwd.mean().item(), "std": fwd.std().item()},
        "forward_backward": {"mean": fwdbwd.mean().item(), "std": fwdbwd.std().item()},
        "full": {"mean": fl.mean().item(), "std": fl.std().item()},
    }

@app.local_entrypoint()
def main():
    model_configs = {
        "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
        "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
        "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
        "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
        "10B":    dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
    }
    warmup_steps = 2
    timed_steps = 10
    print(f"{warmup_steps} warmup steps, {timed_steps} timed steps")
    for model in model_configs:
        result = benchmarking_script.remote(**model_configs[model], w=warmup_steps, n=timed_steps)
        print(f"Model: {model}")
        print(f"Forward only:         Mean ({result['forward']['mean']:.4f}s), Std ({result['forward']['std']:.4f}s)")
        print(f"Forward and backward: Mean ({result['forward_backward']['mean']:.4f}s), Std ({result['forward_backward']['std']:.4f}s)")
        print(f"Full training step:   Mean ({result['full']['mean']:.4f}s), Std ({result['full']['std']:.4f}s)")
        print("")

if __name__ == "__main__":
    main()

"""
Results:

5 warmup steps, 10 timed steps
Model: small
Forward only:         Mean (0.0169s), Std (0.0002s)
Forward and backward: Mean (0.0487s), Std (0.0001s)
Full training step:   Mean (0.0563s), Std (0.0001s)

Model: medium
Forward only:         Mean (0.0477s), Std (0.0004s)
Forward and backward: Mean (0.1393s), Std (0.0001s)
Full training step:   Mean (0.1608s), Std (0.0008s)

Model: large
Forward only:         Mean (0.1063s), Std (0.0001s)
Forward and backward: Mean (0.3133s), Std (0.0002s)
Full training step:   Mean (0.3486s), Std (0.0016s)

Model: xl
Forward only:         Mean (0.2962s), Std (0.0002s)
Forward and backward: Mean (0.8652s), Std (0.0005s)
Full training step:   Mean (0.9447s), Std (0.0002s)

Model: 10B
unable to run due to memory constraints on the GPU

Analysis: Forward + backward pass takes 2.5-3x longer than forward pass alone, indicating that 
the backward pass takes 50-100% longer than forward. The full steps adds a bit of time (~10-15%).
In the medium and large models, standard deviation is comparitively high for full steps, though
this doesn't hold consistent for xl and small. This isn't enough information to differentiate if 
this is a real finding or just noise. Besides that, std dev is generally low across models/steps.

---------------------------------------------------
0 warmup steps, 10 timed steps
Model: small
Forward only:         Mean (0.0385s), Std (0.0698s)
Forward and backward: Mean (0.0680s), Std (0.0631s)
Full training step:   Mean (0.0585s), Std (0.0085s)

Model: medium
Forward only:         Mean (0.0475s), Std (0.0010s)
Forward and backward: Mean (0.1389s), Std (0.0001s)
Full training step:   Mean (0.1558s), Std (0.0019s)

Model: large
Forward only:         Mean (0.1094s), Std (0.0094s)
Forward and backward: Mean (0.3130s), Std (0.0002s)
Full training step:   Mean (0.3424s), Std (0.0004s)

Model: xl
Forward only:         Mean (0.2951s), Std (0.0005s)
Forward and backward: Mean (0.8614s), Std (0.0007s)
Full training step:   Mean (0.9421s), Std (0.0045s)

Model: 10B
unable to run due to memory constraints on the GPU

Analysis: Removing the warmup impacts the small model the most, perhaps because the initial iterations
impact the mean/std dev more than in larger models. In the small model, forward and forward + backward
are significantly higher without warmup. The increase is less apparent in small model's full step and
in other models. Std dev is higher for most of the steps across all models, though the difference is
highest in the small model.

---------------------------------------------------
2 warmup steps, 10 timed steps
Model: small
Forward only:         Mean (0.0164s), Std (0.0000s)
Forward and backward: Mean (0.0481s), Std (0.0002s)
Full training step:   Mean (0.0555s), Std (0.0002s)

Model: medium
Forward only:         Mean (0.0471s), Std (0.0003s)
Forward and backward: Mean (0.1387s), Std (0.0001s)
Full training step:   Mean (0.1558s), Std (0.0017s)

Model: large
Forward only:         Mean (0.1062s), Std (0.0003s)
Forward and backward: Mean (0.3128s), Std (0.0002s)
Full training step:   Mean (0.3424s), Std (0.0002s)

Model: xl
Forward only:         Mean (0.2934s), Std (0.0005s)
Forward and backward: Mean (0.8610s), Std (0.0004s)
Full training step:   Mean (0.9406s), Std (0.0005s)

Model: 10B
unable to run due to memory constraints on the GPU

Analysis: Adding 2 warmup iterations gets the means across all steps/models to around the same as
when there are 5 warmup iterations. Std dev is also similar, though there is a bit of noise where
some scenarios have higher and some have lower std dev than the 5 warmup iteration scenario.
"""