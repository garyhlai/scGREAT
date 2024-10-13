from main import main
import argparse

from our_main import our_main


parser = argparse.ArgumentParser()

# Add hyperparameters as command-line arguments
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--embed_size", type=int, default=768, help="Embedding size")
parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--num_head", type=int, default=4, help="Number of attention heads")
parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
parser.add_argument(
    "--step_size", type=int, default=10, help="Step size for learning rate scheduler"
)
parser.add_argument(
    "--gamma", type=float, default=0.999, help="Gamma value for learning rate scheduler"
)
parser.add_argument(
    "--scheduler_flag", type=bool, default=True, help="Flag to enable/disable scheduler"
)
# get --model_name, default is 'scgreat', but can also be 'gatransformer'
parser.add_argument("--model_name", type=str, default="scgreat", help="Model name")


args = parser.parse_args()

# data_dir = 'mESC500'
data_dir = "hESC500"
print("data_dir:", data_dir)

if args.model_name == "scgreat":
    main(data_dir, args)
elif args.model_name == "gatransformer":
    our_main(data_dir, args)
else:
    raise ValueError("Invalid model name")

print("finish")
