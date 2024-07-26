# general modules
import sys
import os
import numpy as np
from pathlib import Path

# learning framework
import torch
import tqdm
from torch.utils import data as torch_data

from datasets.dataloader import FloodDataset
from utils import evaluation_metrics
from utils import config
from network import model
from utils.loss import soft_dice_loss
from utils.logger import MyWriter

# logging
import wandb


def train(net, cfg, writer:MyWriter):

    # setting device on GPU if available, else CPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # net = torch.nn.DataParallel(net, device_ids=[1, 2, 3])

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=0.0005)

    criterion = soft_dice_loss

    dataset = FloodDataset(train=True, crop_size=cfg.img_size)
    drop_last = True
    batch_size = cfg.batch_size
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": cfg.num_workers,
        "drop_last": drop_last,
        "pin_memory": True,
    }

    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    net_dir = Path(args.output_root) / "run_logs"
    net_dir.mkdir(exist_ok=True)

    positive_pixels = 0
    pixels = 0
    global_step = 0
    epochs = args.epochs
    batches = (
        len(dataset) // batch_size if drop_last else len(dataset) // batch_size + 1
    )

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        loss_tracker = 0
        net.train()
        
        train_thresholds = torch.linspace(0, 1, 101).to(device)
        train_measurer = evaluation_metrics.MultiThresholdMetric(train_thresholds)
        
        for batch in tqdm.tqdm(dataloader):

            t1_img = batch["vv"].to(device)
            t2_img = batch["vh"].to(device)
            rgb_img = batch["rgb"].to(device)

            label = batch["label"].to(device)

            optimizer.zero_grad()

            output = net(t1_img, t2_img)

            loss = criterion(output, label)
            loss_tracker += loss.item()
            loss.backward()
            optimizer.step()
            
            y_pred = torch.sigmoid(output)
            y_true = label.detach()
            y_pred = y_pred.detach()
            train_measurer.add_sample(y_true, y_pred)
            
            positive_pixels += torch.sum(label).item()
            pixels += torch.numel(label)

            global_step += 1
        print(f"Computing Train Indicator ", end=" ", flush=True)
        writer.log_training(loss_tracker / batches,train_measurer.compute_precision.max(),train_measurer.compute_recall.max(),train_measurer.compute_oa.max(),train_measurer.compute_f1.max(),train_measurer.compute_iou.max(),epoch)
        if epoch % 2 == 0:
            print(f"epoch {epoch} / {cfg.epochs}")

            # printing and logging loss
            avg_loss = loss_tracker / batches
            print(f"avg training loss {avg_loss:.5f}")

            # positive pixel ratio used to check oversampling

            wandb.log({f"positive pixel ratio": positive_pixels / pixels})
            positive_pixels = 0
            pixels = 0

            # model evaluation
            # train (different thresholds are tested)
            train_thresholds = torch.linspace(0, 1, 101).to(device)
            train_maxF1, train_maxTresh = model_evaluation(
                net,
                cfg,
                device,
                train_thresholds,
                run_type="Val",
                epoch=epoch,
                step=global_step,
                batches=batches,
                criterion=criterion,
                writer=writer
            )
            # test (using the best training threshold)
            # test_threshold = torch.tensor([train_maxTresh])
            # test_f1, _ = model_evaluation(
            #     net,
            #     cfg,
            #     device,
            #     test_threshold,
            #     run_type="val",
            #     epoch=epoch,
            #     step=global_step,
            # )

        if (epoch + 1) == epochs:
            print(f"saving network", flush=True)
            net_file = net_dir / cfg.name / f"final_net.pkl"
            net_file.parent.mkdir(exist_ok=True)
            torch.save(net.state_dict(), net_file)


def model_evaluation(net, cfg, device, thresholds, run_type, epoch,step, batches,criterion,writer:MyWriter):
    thresholds = thresholds.to(device)
    y_true_set = []
    y_pred_set = []

    measurer = evaluation_metrics.MultiThresholdMetric(thresholds)

    dataset = FloodDataset(train=False, crop_size=cfg.img_size)
    dataloader_kwargs = {
        "batch_size": 1,
        "num_workers": cfg.num_workers,
        "pin_memory": True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    with torch.no_grad():
        net.eval()
        loss_tracker = 0
        for step, batch in enumerate(dataloader):
            t1_img = batch["vv"].to(device)
            t2_img = batch["vh"].to(device)
            y_true = batch["label"].to(device)
            rgb_img = batch["rgb"].to(device)

            y_pred = net(t1_img, t2_img)
            loss = criterion(y_pred, y_true)
            loss_tracker+=loss.item()
            y_pred = torch.sigmoid(y_pred)

            y_true = y_true.detach()
            y_pred = y_pred.detach()
            y_true_set.append(y_true.cpu())
            y_pred_set.append(y_pred.cpu())

            measurer.add_sample(y_true, y_pred)
    
    print(f"Computing {run_type} Train Indicator ", end=" ", flush=True)
    writer.log_images(rgb_img,t1_img,t2_img,y_true, y_pred, epoch)
    f1 = measurer.compute_f1()
    fpr, fnr = measurer.compute_basic_metrics()
    maxF1 = f1.max()
    argmaxF1 = f1.argmax()
    best_fpr = fpr[argmaxF1]
    best_fnr = fnr[argmaxF1]
    best_thresh = thresholds[argmaxF1]
    writer.log_validation(loss_tracker/batches,measurer.compute_precision.max(),measurer.compute_recall.max(),measurer.compute_oa.max(),measurer.compute_f1.max(),measurer.compute_iou.max(),epoch)
    wandb.log(
        {
            f"{run_type} max F1": maxF1,
            f"{run_type} argmax F1": argmaxF1,
            f"{run_type} false positive rate": best_fpr,
            f"{run_type} false negative rate": best_fnr,
            "step": step,
            "epoch": epoch,
        }
    )

    print(f"{maxF1.item():.3f}", flush=True)

    # return maxF1.item(), best_thresh.item()


if __name__ == "__main__":

    os.environ["WANDB_MODE"] = "offline"

    # setting up config based on parsed argument
    parser = config.default_argument_parser()
    args = parser.parse_args()

    # deterministic training
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # loading network
    net = model.create_network(args)

    # tracking land with w&b
    wandb.init(
        name=args.name,
        project="flood_segmentation",
    )
    writer = MyWriter(
        "{}/{}-{}-{}".format(args.log, args.name, args.struct, args.attention)
    )
    # here we go
    try:
        train(net, args, writer)
    except KeyboardInterrupt:
        print("Training terminated")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
