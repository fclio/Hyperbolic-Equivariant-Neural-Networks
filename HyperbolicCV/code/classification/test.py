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
import torch.nn.functional as F
from torch.nn import DataParallel
from os.path import join
import configargparse

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.initialize import select_dataset, select_model, load_model_checkpoint
from lib.utils.visualize import visualize_embeddings
from train import evaluate

from lib.utils.equivariant_test.test_equivariant import *
from lib.utils.utils import AverageMeter, accuracy

def getArguments():
    """ Parses command-line options. """
    parser = configargparse.ArgumentParser(description='Image classification training', add_help=True)

    parser.add_argument('-c', '--config_file', required=False, default=None, is_config_file=True, type=str, 
                        help="Path to config file.")
    parser.add_argument('--exp_name', default="test", type=str,
                        help="Name of the experiment.")
    # Modes
    parser.add_argument('--mode', default="test_accuracy", type=str, 
                        choices=[
                            "test_accuracy",
                            "visualize_embeddings",
                            "fgsm",
                            "pgd",
                            "test_equivairant"
                        ],
                        help = "Select the testing mode.")
    
    # Output settings
    parser.add_argument('--output_dir', default=None, type=str, 
                        help = "Path for output files (relative to working directory).")

    # General settings
    parser.add_argument('--device', default="cuda:0", type=lambda s: [str(item) for item in s.replace(' ','').split(',')],
                        help="List of devices split by comma (e.g. cuda:0,cuda:1), can also be a single device or 'cpu')")
    parser.add_argument('--dtype', default='float32', type=str, choices=["float32", "float64"], 
                        help="Set floating point precision.")
    parser.add_argument('--seed', default=1, type=int, 
                        help="Set seed for deterministic training.")
    parser.add_argument('--load_checkpoint', default=None, type=str, 
                        help = "Path to model checkpoint.")

    # Testing parameters
    parser.add_argument('--batch_size', default=128, type=int, 
                        help="Training batch size.")
    parser.add_argument('--batch_size_test', default=128, type=int, 
                        help="Testing batch size.")
    parser.add_argument('--num_epochs', default=200, type=int,
                        help="Number of training epochs.")
    # Model selection
    parser.add_argument('--num_layers', default=18, type=int, choices=[18, 50], 
                        help = "Number of layers in ResNet.")
    parser.add_argument('--embedding_dim', default=512, type=int, 
                        help = "Dimensionality of classification embedding space (could be expanded by ResNet)")
    parser.add_argument('--encoder_manifold', default='lorentz', type=str, choices=["equivariant","euclidean", "lorentz", "lorentz_equivariant"], 
                        help = "Select conv model encoder manifold.")
    parser.add_argument('--decoder_manifold', default='lorentz', type=str, choices=["equivariant","euclidean", "lorentz", "poincare","lorentz_equivariant"], 
                        help = "Select conv model decoder manifold.")
    
    parser.add_argument('--model_type', default='resnet', type=str,
                        choices=["resnet", "cnn"],
                        help="Select a model type.")
    
    # Hyperbolic geometry settings
    parser.add_argument('--learn_k', action='store_true',
                        help="Set a learnable curvature of hyperbolic geometry.")
    parser.add_argument('--encoder_k', default=1.0, type=float, 
                        help = "Initial curvature of hyperbolic geometry in backbone (geoopt.K=-1/K).")
    parser.add_argument('--decoder_k', default=1.0, type=float, 
                        help = "Initial curvature of hyperbolic geometry in decoder (geoopt.K=-1/K).")
    parser.add_argument('--clip_features', default=1.0, type=float, 
                        help = "Clipping parameter for hybrid HNNs proposed by Guo et al. (2022)")
    
    # Dataset settings
    parser.add_argument('--exp_v', default="", type=str, choices=["v2","v3_1", "v3_2","v3","v4_1", "v4_2","v4", "v5", "v6","v6_1","v2_1","v7","v8"],
                    help="experiment_version")

    # Dataset settings
    parser.add_argument('--dataset', default='CIFAR-100', type=str,
                        choices=["MNIST", "CIFAR-10", "CIFAR-100", "Tiny-ImageNet", "MNIST_rotation", "MNIST_rot", "CIFAR-10_rot", "CIFAR-100_rot","cifar100-lt", "CUB-200"],
                        help="Select a dataset.")

    parser.add_argument('--equivariant_type', default=None, type=str, choices=[ "P4", "P4M"],
                help="Select conv model encoder manifold.")
    args, _ = parser.parse_known_args()

    return args


def save_results_as_json(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

def main(args):
    
    if torch.cuda.is_available():
        device = args.device[0] 
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        print(f"Using CUDA device: {device}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU.")

    results = {}
    print("Arguments:")
    print(args)

    print("Loading dataset...")
    train_loader, val_loader, test_loader, img_dim, num_classes = select_dataset(args, validation_split=True)

    print("Creating model...")
    model = select_model(img_dim, num_classes, args)
    model = model.to(device)
    print('-> Number of model params: {} (trainable: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    if args.load_checkpoint is not None:
        print("Loading model checkpoint from {}".format(args.load_checkpoint))
        model = load_model_checkpoint(model, args.load_checkpoint)
    else:
        print("No model checkpoint given. Using random weights.")

    model = DataParallel(model, device_ids=args.device)
    model.eval()

    if args.mode=="test_accuracy":
        print("Testing accuracy of model...")
        criterion = torch.nn.CrossEntropyLoss()
        loss_test, acc1_test, acc5_test = evaluate(model, test_loader, criterion, device)
        loss_val, acc1_val, acc5_val = evaluate(model, val_loader, criterion, device)

        print(
            f"Validation Results:\n"
            f"  Loss    = {loss_val:.4f}\n"
            f"  Acc@1   = {acc1_val:.4f}\n"
            f"  Acc@5   = {acc5_val:.4f}\n"
            f"Test Results:\n"
            f"  Loss    = {loss_test:.4f}\n"
            f"  Acc@1   = {acc1_test:.4f}\n"
            f"  Acc@5   = {acc5_test:.4f}"
        )


        results["test_accuracy"] = {
            'loss_val': loss_val,
            'acc1_val': acc1_val,
            'acc5_val': acc5_val,
            'loss_test': loss_test,
            'acc1_test': acc1_test,
            'acc5_test': acc5_test
        }
    
        
    elif args.mode=="visualize_embeddings":
        print("Visualizing embedding space of model...")
        if args.output_dir is not None:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            output_path = os.path.join(args.output_dir, "embeddings.png")
        else:
            output_path = "embeddings.png"
        save_embeddings(model, train_loader, output_path, device)

    elif args.mode=="fgsm" or args.mode=="pgd":
        print(f"Attacking model using {args.mode}...")
        attack_results = adversarial_attack(args.mode, model, device, test_loader)

        results["adversarial_attack"] = attack_results

    else:
        print(f"Mode {args.mode} not implemented yet.")

    if args.mode=="test_equivairant":
        path = join("equivariant_test",f"{args.exp_name}-epoch:{args.num_epochs}-{args.exp_v}")
        # test_equivariance(model, test_loader, device, path, args.exp_name )
        tester = EquivarianceTester(model, device, exp_v=args.exp_v, save_path=path,model_type= args.exp_name)
        tester.test_model(test_loader)
    else:
        print("Finished!")
        output_path = os.path.join(args.output_dir, f"{args.exp_name}{args.exp_v}-epoch:{args.num_epochs}_test_{args.dataset}.json")
        save_results_as_json(results, output_path)
    


@torch.no_grad()
def save_embeddings(model, data_loader, output_path, device):
    fig = visualize_embeddings(model, data_loader, device, model.module.dec_manifold, model.module.dec_type=="poincare")
    print(f"Saving embeddings to {output_path}...")
    fig.savefig(output_path)
    plt.close(fig)


def adversarial_attack(attack, model, device, data_loader, epsilons=[0.8/255, 1.6/255, 3.2/255]):
    """ Runs adversarial attacks with different epsilon parameters.
    """
    results = {}
    for eps in epsilons:
        if attack=="fgsm":
            iters=1
        elif attack=="pgd":
            iters=7
        else:
            raise RuntimeError(f"Attack {attack} is not implemented.")
        
        acc1, acc5 = run_attack(attack, model, device, data_loader, eps, iters)
        results[eps] = {"acc@1": acc1, "acc@5": acc5}
    return results
def run_attack(attack, model, device, data_loader, epsilon, iters=7):
    """ Runs adversarial attacks for a single epsilon parameter.
    """
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")

    criterion = torch.nn.CrossEntropyLoss()

    for x, target in tqdm(data_loader):
        x, target = x.to(device), target.to(device)
        x_in = x.data

        for _ in range(iters):
            x.requires_grad = True

            output = model(x)
            init_pred = output.max(1, keepdim=True)[1]

            # Calculate the loss
            loss = criterion(output, target)

            model.zero_grad()
            loss.backward()

            if attack=="fgsm":
                perturbed_img = fgsm_attack(x, epsilon)
            elif attack=="pgd":
                perturbed_img = pgd_attack(x, x_in, epsilon)
            else:
                raise RuntimeError(f"Attack {attack} is not implemented.")
            
            output = model(perturbed_img)
            x = perturbed_img.detach()

        top1, top5 = accuracy(output, target, topk=(1, 5))
        acc1.update(top1.item(), x.shape[0])
        acc5.update(top5.item(), x.shape[0])

    print("Epsilon: {}\tAcc@1={:.4f}, Acc@5={:.4f}".format(epsilon, acc1.avg, acc5.avg))
    return acc1.avg, acc5.avg

def fgsm_attack(x, epsilon=0.3):
    sign_x_grad = x.grad.sign()
    perturbed_img = x + epsilon*sign_x_grad
            
    return perturbed_img

def pgd_attack(x, x_in, epsilon=0.3):
    alpha = epsilon/4.0
    sign_x_grad = x.grad.sign()
    x = x + alpha*sign_x_grad
    eta = torch.clamp(x - x_in, min=-epsilon, max=epsilon)
    perturbed_img = x_in + eta
            
    return perturbed_img

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