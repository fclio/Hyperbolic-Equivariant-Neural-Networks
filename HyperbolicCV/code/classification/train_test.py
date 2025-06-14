# -----------------------------------------------------
# Change working directory to parent HyperbolicCV/code
import os
import sys

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
# -----------------------------------------------------
import json
import torch
from torch.nn import DataParallel

import configargparse
from tqdm import tqdm

import random
import numpy as np

from utils.initialize import select_dataset, select_model, select_optimizer, load_checkpoint
from lib.utils.utils import AverageMeter, accuracy


def getArguments():
    """ Parses command-line options. """
    parser = configargparse.ArgumentParser(description='Image classification training', add_help=True)

    parser.add_argument('-c', '--config_file', required=False, default=None, is_config_file=True, type=str,
                        help="Path to config file.")

    # Output settings
    parser.add_argument('--exp_name', default="test", type=str,
                        help="Name of the experiment.")
    parser.add_argument('--output_dir', default=None, type=str,
                        help="Path for output files (relative to working directory).")

    # General settings
    parser.add_argument('--device', default="cuda:0",
                        type=lambda s: [str(item) for item in s.replace(' ', '').split(',')],
                        help="List of devices split by comma (e.g. cuda:0,cuda:1), can also be a single device or 'cpu')")
    parser.add_argument('--dtype', default='float32', type=str, choices=["float32", "float64"],
                        help="Set floating point precision.")
    parser.add_argument('--seed', default=1, type=int,
                        help="Set seed for deterministic training.")
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help="Path to model checkpoint (weights, optimizer, epoch).")
    parser.add_argument('--compile', action='store_true',
                        help="Compile model for faster inference (requires PyTorch 2).")

    # General training parameters
    parser.add_argument('--num_epochs', default=200, type=int,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="Training batch size.")
    parser.add_argument('--lr', default=1e-1, type=float,
                        help="Training learning rate.")
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help="Weight decay (L2 regularization)")
    parser.add_argument('--optimizer', default="RiemannianSGD", type=str,
                        choices=["RiemannianAdam", "RiemannianSGD", "Adam", "SGD"],
                        help="Optimizer for training.")
    parser.add_argument('--use_lr_scheduler', action='store_true',
                        help="If learning rate should be reduced after step epochs using a LR scheduler.")
    parser.add_argument('--lr_scheduler_milestones', default=[60, 120, 160], type=int, nargs="+",
                        help="Milestones of LR scheduler.")
    parser.add_argument('--lr_scheduler_gamma', default=0.2, type=float,
                        help="Gamma parameter of LR scheduler.")

    # General validation/testing hyperparameters
    parser.add_argument('--batch_size_test', default=128, type=int,
                        help="Validation/Testing batch size.")

    # Model selection
    parser.add_argument('--num_layers', default=18, type=int, choices=[18, 50],
                        help="Number of layers in ResNet.")
    parser.add_argument('--embedding_dim', default=512, type=int,
                        help="Dimensionality of classification embedding space (could be expanded by ResNet)")
    parser.add_argument('--encoder_manifold', default='lorentz', type=str, choices=["euclidean", "lorentz", "equivariant", "lorentz_equivariant" ],
                        help="Select conv model encoder manifold.")
    parser.add_argument('--decoder_manifold', default='lorentz', type=str, choices=["euclidean", "lorentz", "equivariant","lorentz_equivariant","poincare"],
                        help="Select conv model decoder manifold.")

    # Hyperbolic geometry settings
    parser.add_argument('--learn_k', action='store_true',
                        help="Set a learnable curvature of hyperbolic geometry.")
    parser.add_argument('--encoder_k', default=1.0, type=float,
                        help="Initial curvature of hyperbolic geometry in backbone (geoopt.K=-1/K).")
    parser.add_argument('--decoder_k', default=1.0, type=float,
                        help="Initial curvature of hyperbolic geometry in decoder (geoopt.K=-1/K).")
    parser.add_argument('--clip_features', default=1.0, type=float,
                        help="Clipping parameter for hybrid HNNs proposed by Guo et al. (2022)")

    parser.add_argument('--model_type', default='resnet', type=str,
                        choices=["resnet", "cnn"],
                        help="Select a model type.")
    
    parser.add_argument('--equivariant_type', default=None, type=str, choices=[ "P4", "P4M"],
                    help="Select conv model encoder manifold.")
    

    parser.add_argument('--exp_v', default="", type=str, choices=["v2","v3_1", "v3_2","v3","v4_1", "v4_2","v4", "v5", "v6","v6_1","v2_1","v2_2","v2_3","v7","v8"],
                    help="experiment_version")

    # Dataset settings
    parser.add_argument('--dataset', default='CIFAR-100', type=str,
                        choices=["MNIST", "CIFAR-10", "CIFAR-100", "Tiny-ImageNet", "MNIST_rotation", "MNIST_rot", "CIFAR-10_rot", "CIFAR-100_rot","cifar100-lt", "CUB-200", "cifar10-lt","Flower102","Food101","CelebA","iNaturalist", "LFWPeople","PCAM"],
                        help="Select a dataset.")
    


    args = parser.parse_args()

    return args


def save_test_results(results, output_dir):
    """ Saves the test results to a JSON file. """
    results_file = os.path.join(output_dir, f"test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Test results saved to {results_file}")

def main(args):

    # Automatically detect device
    if torch.cuda.is_available():
        device = args.device[0] 
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        print(f"Using CUDA device: {device}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU.")

    print("Running experiment: " + args.exp_name)

    print("Arguments:")
    print(args)

    output_dir = os.path.join(args.output_dir, f"{args.exp_name}_{args.dataset}")

    if output_dir is not None:
        if not os.path.exists(output_dir):
            print("Creating missing output directory...")
            os.makedirs(output_dir, exist_ok=True) 

    print("Loading dataset...")
    train_loader, val_loader, test_loader, img_dim, num_classes = select_dataset(args)

    print("Creating model...")
    model = select_model(img_dim, num_classes, args)
    model = model.to(device)
    print('-> Number of model params: {} (trainable: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    print("Creating optimizer...")
    optimizer, lr_scheduler = select_optimizer(model, args)
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    if args.load_checkpoint is not None:
        print("Loading model checkpoint from {}".format(args.load_checkpoint))
        model, optimizer, lr_scheduler, start_epoch = load_checkpoint(model, optimizer, lr_scheduler, args)

    model = DataParallel(model, device_ids=args.device)
    for name, param in model.named_parameters():
        print(f"{name}: {param.data.size()}")
    for name, param in model.named_parameters():
        print(f'{name}: requires_grad={param.requires_grad}')
        print(f"{name}: {param.data.size()}")


    if args.compile:
        model = torch.compile(model)

    print("Training...")
    global_step = start_epoch * len(train_loader)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        losses = AverageMeter("Loss", ":.4e")
        acc1 = AverageMeter("Acc@1", ":6.2f")
        acc5 = AverageMeter("Acc@5", ":6.2f")

        for i, (x, y) in tqdm(enumerate(train_loader)):
            # ------- Start iteration -------
            x = x.to(device)
            y = y.to(device)
            print("starting")
            logits = model(x)
            print("y",y)
            print("logits",logits)
            loss = criterion(logits, y)
            print("loss",loss)

            before_weights = {name: param.clone() for name, param in model.named_parameters()}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                if torch.equal(before_weights[name], param):
                    print(f"{name} did not update")
                else:
                    print(f"{name} updated correctly: {param.data.shape}")

                if param.grad is not None:
                    print(f'{name} gradients: {param.grad.shape}')
                else:
                    print(f'{name} has no gradients.')
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of {name}")

            if i == 4:
                rde

                
            with torch.no_grad():
                top1, top5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item())
                acc1.update(top1.item())
                acc5.update(top5.item())

            global_step += 1
       
            # ------- End iteration -------
            print("lose list", losses)
        # ------- Start validation and logging -------
        with torch.no_grad():
            if lr_scheduler is not None:
                if (epoch + 1) == args.lr_scheduler_milestones[0]:  # skip the first drop for some Parameters
                    optimizer.param_groups[1]['lr'] *= (1 / args.lr_scheduler_gamma) # Manifold params
                    print("Skipped lr drop for manifold parameters")

                lr_scheduler.step()

            loss_val, acc1_val, acc5_val = evaluate(model, val_loader, criterion, device)

            print(
                "Epoch {}/{}: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}, Validation: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
                    epoch + 1, args.num_epochs, losses.avg, acc1.avg, acc5.avg, loss_val, acc1_val, acc5_val))
       
            # Testing for best model
            if acc1_val > best_acc:
                best_acc = acc1_val
                best_epoch = epoch + 1
                if args.output_dir is not None:
                    save_path = output_dir + "/best_model.pth"
                    torch.save({
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        'epoch': epoch,
                        'args': args,
                    }, save_path)
        # ------- End validation and logging -------

    print("-----------------\nTraining finished\n-----------------")
    print("Best epoch = {}, with Acc@1={:.4f}".format(best_epoch, best_acc))

    if args.output_dir is not None:
        save_path = output_dir + "/final_model.pth" 
  
        torch.save({
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'epoch': epoch,
            'args': args,
        }, save_path)
        print("Model saved to " + save_path)
    else:
        print("Model not saved.")

    print("Testing final model...")
    loss_test, acc1_test, acc5_test = evaluate(model, test_loader, criterion, device)

    print("Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
        loss_test, acc1_test, acc5_test))

    print("Testing best model...")
    if args.output_dir is not None:
        print("Loading best model...")
        save_path = output_dir + "/final_model.pth" 
        checkpoint = torch.load(save_path, map_location=device)
        model.module.load_state_dict(checkpoint['model'], strict=True)

        loss_test, acc1_test, acc5_test = evaluate(model, test_loader, criterion, device)

        test_results = {
            'loss_test': loss_test,
            'acc1_test': acc1_test,
            'acc5_test': acc5_test
        }
        print(f"Final Model Results: Loss={loss_test:.4f}, Acc@1={acc1_test:.4f}, Acc@5={acc5_test:.4f}")

        # Save test results to JSON
        if args.output_dir:
            save_test_results(test_results, output_dir)

    else:
        print("Best model not saved, because no output_dir given.")


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """ Evaluates model performance """
    model.eval()
    model.to(device)

    losses = AverageMeter("Loss", ":.4e")
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(logits, y)

        top1, top5 = accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item())
        acc1.update(top1.item(), x.shape[0])
        acc5.update(top5.item(), x.shape[0])

    return losses.avg, acc1.avg, acc5.avg 


# ----------------------------------
if __name__ == '__main__':
    args = getArguments()

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise "Wrong dtype in configuration -> " + args.dtype

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    main(args)
    