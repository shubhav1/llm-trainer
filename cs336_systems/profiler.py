import torch
# from torch.profiler import profile, record_function, ProfilerActivity
import torch.cuda.nvtx as nvtx
# import modal
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

# app = modal.App("cs336-profiling")
# image = (
#     modal.Image.debian_slim(python_version="3.12")
#     .pip_install("torch~=2.11.0", "einops>=0.8", "einx>=0.4", "jaxtyping>=0.3", "numpy>=2.4")
#     .add_local_python_source("cs336_basics")
# )

# @app.function(image=image, gpu="B200", timeout=1800)
def profile_model(d_model, d_ff, num_layers, num_heads, w, n, context_length):
    """
    A script that will initialize a basics Transformer model with the given
    hyperparameters, create a random batch of data, and run torch profiler on 
    forward pass, backward pass, and the optimizer step
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize model based on given hyperparameters
    model = BasicsTransformerLM(vocab_size=10000, context_length=context_length, 
                                d_model=d_model, num_layers=num_layers, 
                                num_heads=num_heads, d_ff=d_ff).to(device)

    # generate random batch of data
    batch_size = 4
    vocab_size = 10000
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    optimizer = AdamW(model.parameters()) 

    # warmup steps
    for _ in range(w):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize() if device == "cuda" else None

    # profiled steps, nvtx-labeled so nsys can isolate each region
    for step in range(n):
        with nvtx.range(f"step_{step}"):
            optimizer.zero_grad()
            with nvtx.range("forward_pass"):
                logits = model(x)
            with nvtx.range("backward_pass"):
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
            with nvtx.range("optimizer_step"):
                optimizer.step()
        torch.cuda.synchronize() if device == "cuda" else None


# @app.local_entrypoint()
def main():
    model_configs = {
        "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
        "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
        "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    }

    warmup_steps = 5
    profiled_steps = 5
    context_length = 512

    print(f"{warmup_steps} warmup steps, {profiled_steps} profiled step(s)")
    for model in model_configs:
        print(f"Profiling model: {model}")
        profile_model(**model_configs[model], w=warmup_steps, n=profiled_steps, context_length=context_length)
        print(f"Finished profiling model: {model}")

if __name__ == "__main__":
    main()

"""
Results:


"""