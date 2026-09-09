import torch
import os

from model import NameGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Load

def load(path="saved_models/best_model.pt"):
    ckpt=torch.load(path, map_location=device, weights_only=False)
    cfg=ckpt["config"]
    itos={int(k): v for k, v in ckpt["vocab"]["itos"].items()}
    vocab_size=ckpt['vocab']["vocab_size"]
    
    model = NameGenerator(
        vocab_size = vocab_size,
        n_embd     = cfg["n_embd"],
        block_size = cfg["block_size"],
        n_hidden   = cfg["n_hidden"],
        n_layers   = cfg["n_layers"],
        dropout    = cfg["dropout"]
    )
    
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded  (val loss: {ckpt['result']['best_val_loss']:.4f})")
    return model, cfg, itos


### Console app

def run():
    print("\n" + "="*50)
    print("  Name Generator")
    print("="*50)

    model, cfg, itos = load()

    print("\nCommands:")
    print("  generate <n> <temp>   e.g.  generate 10 0.8")
    print("  compare               low / mid / high temperature")
    print("  info                  model details")
    print("  quit\n")
    
    while True:
        try:
            raw=input(">>>").strip()
            
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not raw:
            continue
        
        parts = raw.split()
        cmd   = parts[0].lower()
        
        if cmd == "quit":
            print("Goodbye!")
            break
        
        elif cmd == "info":
            print("\nConfig:")
            for k, v in cfg.items():
                print(f"  {k:<15} {v}")
            total = sum(p.numel() for p in model.parameters())
            print(f"  {'parameters':<15} {total:,}")
            print(f"  {'device':<15} {device}\n")
            
        elif cmd == "generate":
            n    = int(parts[1])   if len(parts) > 1 else 10
            temp = float(parts[2]) if len(parts) > 2 else 0.8
            out  = model.generate(n, temp, itos)
            print(f"\nGenerated {n} names (temp={temp}):")
            for i, name in enumerate(out, 1):
                print(f"  {i:2d}. {name.capitalize()}")
            print()
            
        elif cmd == "compare":
            print()
            for label, t in [("conservative", 0.4), ("balanced", 0.8), ("creative", 1.4)]:
                out = model.generate(5, t, itos)
                print(f"  [{label:<13} {t}]  {',  '.join(n.capitalize() for n in out)}")
            print()

        else:
            print(f"  Unknown: '{cmd}'\n")



if __name__ == "__main__":
    run()

  
    
