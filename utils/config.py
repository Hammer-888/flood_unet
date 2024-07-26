import argparse


def default_argument_parser():

    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument("--img_size", type=int, default=224, help="input image size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="seed for initializing training."
    )
    parser.add_argument(
        "--name", default="flood-experiment", type=str, help="name of the experiment"
    )
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    parser.add_argument(
        "--output_root", default="./outputs", type=str, help="output root"
    )
    # parser.add_argument(
    #     "--model", default="dualstreamunet", type=str, help="name of the model"
    # )
    parser.add_argument(
        "--struct",
        default="unet",
        type=str,
        help="name of the structure,unet or resunet",
    )
    parser.add_argument("--attention", default=True, type=bool, help="use attention")
    parser.add_argument(
        "--topology", default=[64, 128, 256, 512], type=list, help="model topology"
    )
    parser.add_argument("--n_classes", default=1, type=int, help="number of classes")
    parser.add_argument(
        "--in_channels", default=1, type=int, help="number of input channels"
    )
    parser.add_argument("--log", type=str, default="logs", help="logs directory")
    return parser
