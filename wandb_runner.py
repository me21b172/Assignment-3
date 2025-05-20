import argparse
import wandb
import torch
from torch.utils.data import DataLoader
from utils import collate_fn, LangDataset,devnagri2int,latinList2int
from model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from train import train_model

SEED = 42
g = torch.Generator()
g.manual_seed(SEED)
def sweep_config(best_config=False):
    """Define the configuration for hyperparameter sweep"""
    base_params = {
        "embed_size": {"values": [256, 512]},
        "num_layers": {"values": [ 3, 4]},
        "layer": {"values": ["lstm", "gru"]},
        "hidden_size": {"values": [256, 512]},
        "batch_size": {"values": [32, 64]},
        "learning_rate": {"values": [5e-3, 1e-3, 1e-2]},
        "dropout_p": {"values": [0.1, 0.3]},
        "activation": {"values": ["tanh"]},
        "teacher_forcing_prob": {"values": [ 0.8, 0.9, 0.99]},
        "beam_width": {"values": [4, 5]},
        "num_epochs": {"values": [4]}
    }

    if not best_config:
        return {
            "method": "bayes",
            "metric": {"name": "val_accuracy", "goal": "maximize"},
            "parameters": base_params
        }
    else:
        # Fix to best-known values
        best_params = {
            "embed_size": {"values": [512]},
            "num_layers": {"values": [3]},
            "layer": {"values": ["lstm"]},
            "hidden_size": {"values": [512]},
            "batch_size": {"values": [64]},
            "learning_rate": {"values": [1e-3]},
            "dropout_p": {"values": [0.1]},
            "activation": {"values": ["tanh"]},
            "teacher_forcing_prob": {"values": [0.9]},
            "beam_width": {"values": [5]},
            "num_epochs": {"values": [6]}
        }
        return {
            "method": "bayes",
            "metric": {"name": "test_accuracy", "goal": "maximize"},
            "parameters": best_params
        }


def wandb_train(attention=False, best_config= False):
    """Main training function for a wandb run"""
    # Initialize wandb
    run = wandb.init()
    config = run.config
    run = wandb.init()
    config = run.config
    run.name = f"Layer-{config.layer}-Batch-{config.batch_size}-LR-{config.learning_rate}-Dropout-{config.dropout_p}-Layers-{config.num_layers}-LayerType-{config.layer}-BeamWidth-{config.beam_width}"
    run.save()
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data loaders using config.batch_size
    dataset = "test" if best_config else "val"
    print(dataset)
    train_dataset,val_dataset = LangDataset("train"),LangDataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn, generator=g)

    # Instantiate models
    encoder = EncoderRNN(
        vocab_size=len(latinList2int),
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        layer=config.layer,
        dropout_p=config.dropout_p
    ).to(device)

    if attention:
        """Attention Decoder"""
        print("YEs Attention!!")
        decoder = AttnDecoderRNN(
            vocab_size=len(devnagri2int),
            embed_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            layer=config.layer
        ).to(device)
    else:
        decoder = DecoderRNN(
            vocab_size=len(devnagri2int),
            embed_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            layer=config.layer
        ).to(device)

    # Call your training loop
    train_model(train_loader, val_loader,encoder, decoder, 
          n_epochs=config.num_epochs, learning_rate=config.learning_rate,
          teacher_forcing_prob=config.teacher_forcing_prob,beam_width=config.beam_width,
          print_every=1, plot_every=10,iswandb=True,best_config=best_config,attention=attention)
    # Finish wandb run
    run.finish()


def run_sweep(args,sweep_id=None):
    """Create or run a wandb sweep."""
    if args.wandb_sweeps:
        if sweep_id is None:
            print(f"best congif: {args.best_config}")
            sweep_id = wandb.sweep(sweep_config(args.best_config), project="transliteration-sweep")
        wandb.agent(sweep_id, function = lambda: wandb_train(args.attention,args.best_config), count=1 if args.best_config else 15)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataset,val_dataset = LangDataset("train"),LangDataset("val")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn)

        # Instantiate models
        encoder = EncoderRNN(
            vocab_size=len(latinList2int),
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            layer=args.layer,
            dropout_p=args.dropout_p
        ).to(device)
        if args.attention:
            decoder = AttnDecoderRNN(
                vocab_size=len(devnagri2int),
                embed_size=args.embed_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                layer=args.layer
            ).to(device)
        else:
            decoder = DecoderRNN(
                vocab_size=len(devnagri2int),
                embed_size=args.embed_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                layer=args.layer
            ).to(device)

    # Call your training loop
        train_model(train_loader, val_loader,encoder, decoder, 
            n_epochs=args.num_epochs, learning_rate=args.learning_rate,
            teacher_forcing_prob=args.teacher_forcing_prob,beam_width=args.beam_width,
            print_every=1, plot_every=10,iswandb=False)
        return encoder,decoder

if __name__ == "__main__":
    # take arguments from command line
    parser = argparse.ArgumentParser(description="Run a wandb sweep for hyperparameter tuning.")
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID to run.")
    parser.add_argument("--best_config", dest='best_config', action='store_true', help='runs sweep with best parameters')
    parser.add_argument('--attention', dest='attention', action='store_true', help='runs Attn Decoder')
    parser.add_argument('--wandb_sweeps', dest='wandb_sweeps', action='store_true', help='want wandb swweps to tune parameters?')
    parser.add_argument("--embed_size", type=int, default=256, help="Size of embedding vector")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the RNN")
    parser.add_argument("--layer", type=str, default="lstm", help="Type of RNN layer (lstm, rnn, gru)")
    parser.add_argument("--hidden_size", type=int, default=256, help="Size of hidden state")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout_p", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--activation", type=str, default="tanh", help="Activation function")
    parser.add_argument("--teacher_forcing_prob", type=float, default=0.5, help="Teacher forcing probability")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam width for beam search")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()
    run_sweep(args=args)