import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.utils.data import ConcatDataset
from lib.geoopt import ManifoldParameter
from lib.geoopt.optim import RiemannianAdam, RiemannianSGD
from torch.optim.lr_scheduler import MultiStepLR
import os
from classification.models.classifier_resnet import ResNetClassifier

from classification.models.classifier_cnn_experiment import CNNClassifier

from classification.utils.dataset import Dataset, CIFAR100LT, save_image, CIFAR10LT, TransformedSubset




def load_checkpoint(model, optimizer, lr_scheduler, device, args):
    """ Loads a checkpoint from file-system. """

    if 'weights_only' in torch.load.__code__.co_varnames:
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)

    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    if 'optimizer' in checkpoint:
        if checkpoint['args'].optimizer == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # for group in optimizer.param_groups:
            #     group['lr'] = args.lr

            if (lr_scheduler is not None) and ('lr_scheduler' in checkpoint):
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            print("Warning: Could not load optimizer and lr-scheduler state_dict. Different optimizer in configuration ({}) and checkpoint ({}).".format(args.optimizer, checkpoint['args'].optimizer))

    epoch = 0
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {epoch}")

    return model, optimizer, lr_scheduler, epoch

def load_model_checkpoint(model, checkpoint_path, map_location='cpu'):
    """ Loads a checkpoint from file-system. """

    if 'weights_only' in torch.load.__code__.co_varnames:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    return model

def select_model(img_dim, num_classes, args):
    """ Selects and sets up an available model and returns it. """

    if args.model_type == "resnet":
        enc_args = {
            'img_dim' : img_dim,
            'embed_dim' : args.embedding_dim,
            'num_classes' : num_classes,
            'bias' : args.encoder_manifold=="lorentz"
        }

        if args.encoder_manifold=="lorentz":
            enc_args['learn_k'] = args.learn_k
            enc_args['k'] = args.encoder_k
        elif args.encoder_manifold=="equivariant":
            enc_args["eq_type"] = args.equivariant_type

        dec_args = {
            'embed_dim' : args.embedding_dim,
            'num_classes' : num_classes,
            'k' : args.decoder_k,
            'learn_k' : args.learn_k,
            'type' : 'mlr',
            'clip_r' : args.clip_features
        }

        model = ResNetClassifier(
            num_layers=args.num_layers,
            enc_type=args.encoder_manifold,
            dec_type=args.decoder_manifold,
            enc_kwargs=enc_args,
            dec_kwargs=dec_args
        )

    elif args.model_type == "cnn":
        enc_args = {
            'img_dim' : img_dim,
            'embed_dim' : args.embedding_dim,
            'num_classes' : num_classes,
            # 'bias' : args.encoder_manifold=="lorentz"
        }

        if args.encoder_manifold=="lorentz":
            enc_args['learn_k'] = args.learn_k
            enc_args['k'] = args.encoder_k
        elif args.encoder_manifold=="equivariant":
            enc_args["eq_type"] = args.equivariant_type

        elif args.encoder_manifold=="lorentz_equivariant":
            enc_args['learn_k'] = args.learn_k
            enc_args['k'] = args.encoder_k
            enc_args["eq_type"] = args.equivariant_type
            enc_args["exp_v"] = args.exp_v

        dec_args = {
            'embed_dim' : args.embedding_dim,
            'num_classes' : num_classes,
            'k' : args.decoder_k,
            'learn_k' : args.learn_k,
            'type' : 'mlr',
            'clip_r' : args.clip_features
        }

        model = CNNClassifier(
            enc_type=args.encoder_manifold,
            dec_type=args.decoder_manifold,
            cnn_size=args.cnn_size,
            enc_kwargs=enc_args,
            dec_kwargs=dec_args
        )
    return model

def select_optimizer(model, args):
    """ Selects and sets up an available optimizer and returns it. """

    model_parameters = get_param_groups(model, args.lr*args.lr_scheduler_gamma, args.weight_decay)

    if args.optimizer == "RiemannianAdam":
        optimizer = RiemannianAdam(model_parameters, lr=args.lr, weight_decay=args.weight_decay, stabilize=1)
    elif args.optimizer == "RiemannianSGD":
        optimizer = RiemannianSGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, stabilize=1)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model_parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    else:
        raise "Optimizer not found. Wrong optimizer in configuration... -> " + args.model

    lr_scheduler = None
    if args.use_lr_scheduler:
        lr_scheduler = MultiStepLR(
            optimizer, milestones=args.lr_scheduler_milestones, gamma=args.lr_scheduler_gamma
        )


    return optimizer, lr_scheduler

def get_param_groups(model, lr_manifold, weight_decay_manifold):
    no_decay = ["scale"]
    k_params = ["manifold.k"]

    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_decay)
                and not isinstance(p, ManifoldParameter)
                and not any(nd in n for nd in k_params)
            ],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and isinstance(p, ManifoldParameter)
            ],
            'lr' : lr_manifold,
            "weight_decay": weight_decay_manifold
        },
        {  # k parameters
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and any(nd in n for nd in k_params)
            ],
            "weight_decay": 0,
            "lr": 1e-4
        }
    ]

    return parameters

def select_dataset(args, validation_split=True,data_dir = '/projects/prjs1590/data'):
    """ Selects an available dataset and returns PyTorch dataloaders for training, validation and testing. """

    if args.dataset == 'MNIST':

        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

        img_dim = [1, 32, 32]
        num_classes = 10

    elif args.dataset == 'MNIST_rotation':

        train_transform=transforms.Compose([
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        test_transform=transforms.Compose([
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

        img_dim = [1, 32, 32]
        num_classes = 10

    elif args.dataset == 'MNIST_rot':

        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        test_transform=transforms.Compose([
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

        img_dim = [1, 32, 32]
        num_classes = 10

    elif args.dataset == 'CIFAR-10':
        train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

        img_dim = [3, 32, 32]
        num_classes = 10

    elif args.dataset == 'CIFAR-10_rot':
        train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        test_transform=transforms.Compose([
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

        img_dim = [3, 32, 32]
        num_classes = 10

    elif args.dataset == 'CIFAR-100':
        train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)


        img_dim = [3, 32, 32]
        num_classes = 100

    elif args.dataset == 'CIFAR-100_rot':
        train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        test_transform=transforms.Compose([
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

        img_dim = [3, 32, 32]
        num_classes = 100

    elif args.dataset == 'Tiny-ImageNet':
        root_dir = os.path.join( data_dir, "tiny-imagenet-200")
        train_dir = os.path.join( root_dir , "train/images")
        val_dir = os.path.join(root_dir , "val/images")
        test_dir = os.path.join(root_dir , "val/images") # TODO: No labels for test were given, so treat validation as test
        print(train_dir)
        train_transform=transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform=transforms.Compose([
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_set = datasets.ImageFolder(train_dir, train_transform)
        val_set = datasets.ImageFolder(val_dir, val_transform)
        test_set = datasets.ImageFolder(test_dir, test_transform)

        img_dim = [3, 64, 64]
        num_classes = 200

    elif args.dataset == 'cifar100-lt':

        # Define your transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        test_transform = transforms.Compose([
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        # Load the dataset with the desired imbalance factor
        # Options for config_name: 'r-10' (imbalance factor 10), 'r-100' (imbalance factor 100)
        dataset = load_dataset("tomas-gajarsky/cifar100-lt", name="r-100",cache_dir=data_dir)

        # Create training and test datasets
        train_set = CIFAR100LT(dataset['train'], transform=train_transform)
        test_set = CIFAR100LT(dataset['test'], transform=test_transform)

        # Optionally, create a validation split
        if validation_split:
            # train_set, val_set = torch.utils.data.random_split(train_set, [40000, len(train_set) - 40000], generator=torch.Generator().manual_seed(1))
            val_set =  CIFAR100LT(dataset['test'], transform=train_transform)



        img_dim = [3, 32, 32]
        num_classes = 100

    elif args.dataset == 'cifar10-lt':

        # Define your transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        test_transform = transforms.Compose([
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        dataset = load_dataset("tomas-gajarsky/cifar10-lt", name="r-10",cache_dir=data_dir)

        # Create training and test datasets
        train_set = CIFAR10LT(dataset['train'], transform=train_transform)
        test_set = CIFAR10LT(dataset['test'], transform=test_transform)

        # Optionally, create a validation split
        if validation_split:
            # train_set, val_set = torch.utils.data.random_split(train_set, [40000, len(train_set) - 40000], generator=torch.Generator().manual_seed(1))
            val_set =  CIFAR10LT(dataset['test'], transform=train_transform)
        img_dim = [3, 32, 32]
        num_classes = 10

    elif args.dataset == 'CUB-200':

        img_dim = [3, 224, 224]
        # img_dim = [3, 32, 32]
        train_transform=transforms.Compose([
            transforms.Resize(img_dim[1:]),
            transforms.RandomCrop(img_dim[1], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        val_transform=transforms.Compose([
            transforms.Resize(img_dim[1:]),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        test_transform=transforms.Compose([
            transforms.Resize(img_dim[1:]),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])
        dataset = load_dataset("Mobulan/CUB-200-2011",cache_dir=data_dir)
        # dataset = load_dataset("Donghyun99/CUB-200-2011")
        # dataset = load_dataset("GATE-engine/cubirds200")
        # train_set = Dataset(dataset['train'], transform=train_transform)
        # test_set = Dataset(dataset['test'], transform=test_transform)
        # train_set = CUB200Dataset(dataset['test'], transform=test_transform)

        # Split raw HuggingFace dataset first
        full_train_data = dataset['train']
        train_size = int(0.9 * len(full_train_data))
        val_size = len(full_train_data) - train_size
        train_data, val_data = full_train_data.train_test_split(test_size=val_size, seed=42).values()

        # Now apply transforms separately
        train_set = Dataset(train_data, transform=train_transform)
        test_set = Dataset(val_data, transform=test_transform)

        if validation_split:
            # val_set = Dataset(dataset['validation'], transform=val_transform)
            val_set = Dataset(val_data, transform=val_transform)
        save_image(train_set[0][0],args.output_dir)

        num_classes = 200

    elif args.dataset == 'Flower102':

        # Flower102 images are typically 224x224 (or larger)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),   # crop and resize to 224x224
            transforms.RandomHorizontalFlip(),    # augmentation
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],       # standard ImageNet mean/std (common for Flowers)
                std=[0.229, 0.224, 0.225]
            ),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),               # resize shorter side to 256
            transforms.CenterCrop(224),           # crop center 224x224
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),               # resize shorter side to 256
            transforms.CenterCrop(224),           # crop center 224x224
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        train_set = datasets.Flowers102(root=data_dir, split='train', download=True, transform=train_transform)
        val_set = datasets.Flowers102(root=data_dir, split='val', download=True, transform=test_transform)
        test_set = datasets.Flowers102(root=data_dir, split='test', download=True, transform=test_transform)

        img_dim = [3, 224, 224]
        num_classes = 102

    elif args.dataset == 'Food101':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),      # Food101 images are larger, usually 224x224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],          # Standard ImageNet normalization
                std=[0.229, 0.224, 0.225]
            ),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],          # Same normalization for test
                std=[0.229, 0.224, 0.225]
            ),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],          # Same normalization for test
                std=[0.229, 0.224, 0.225]
            ),
        ])


        train_set = datasets.Food101(root=data_dir, split='train', download=True, transform=train_transform)
        val_set = datasets.Food101(root=data_dir, split='test', download=True, transform=val_transform)
        test_set = datasets.Food101(root=data_dir, split='test', download=True, transform=test_transform)

        img_dim = [3, 224, 224]
        num_classes = 101

    elif args.dataset == 'CelebA':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],       # CelebA images normalized between -1 and 1 roughly
                std=[0.5, 0.5, 0.5]
            ),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])

        train_set = datasets.CelebA(root=data_dir, split='valid', download=True, transform=train_transform)
        val_set = datasets.CelebA(root=data_dir, split='test', download=True, transform=val_transform)
        test_set = datasets.CelebA(root=data_dir, split='test', download=True, transform=test_transform)

        img_dim = [3, 224, 224]
        num_classes = 40   # CelebA has 40 attribute labels (not classic classes)

    elif args.dataset == 'iNaturalist':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet stats (commonly used for iNaturalist too)
                std=[0.229, 0.224, 0.225]
            ),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        full_dataset = datasets.INaturalist(
            root=data_dir,
            version='2021_valid',
            download=True,
            transform=None,
        )

        # Decide split sizes
        total_len = len(full_dataset)
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len

        # Random split indices
        train_indices, val_indices = torch.utils.data.random_split(
            list(range(total_len)), [train_len, val_len]
        )

        # Create train and val datasets with their respective transforms
        train_set = TransformedSubset(full_dataset, train_indices, train_transform)
        val_set = TransformedSubset(full_dataset, val_indices, val_transform)
        test_set = TransformedSubset(full_dataset, val_indices, test_transform)

        print(len(full_dataset.all_categories))  # Number of classes

        img_dim = [3, 224, 224]
        num_classes = 10000  # iNaturalist 2019 has 1010 classes

    elif args.dataset == 'PCAM':
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])

            val_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.RandomRotation(360),
                transforms.ToTensor(),
            ])

            train_set = datasets.PCAM(root=data_dir, split='train', transform=train_transform, download=True)
            # if validation_split:
            #     # You can split 70k train -> 60k train / 10k val (or adapt sizes to your memory/quota)
            #     train_set, val_set = torch.utils.data.random_split(
            #         train_set,
            #         [60000, len(train_set) - 60000],
            #         generator=torch.Generator().manual_seed(1)
            #     )
            # else:

            val_set = datasets.PCAM(root=data_dir, split='val', transform=val_transform, download=True)

            test_set = datasets.PCAM(root=data_dir, split='test', transform=test_transform, download=True)

            img_dim = [3, 96, 96]
            num_classes = 2  # Binary classification: tumor vs non-tumor
    elif args.dataset == 'SUN397':

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Load full dataset (download=True if not downloaded)
        full_dataset = datasets.SUN397(root=data_dir, transform=None, download=True)

        # Decide split sizes
        total_len = len(full_dataset)
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len

        # Random split indices
        train_indices, val_indices = torch.utils.data.random_split(
            list(range(total_len)), [train_len, val_len]
        )

        # Create train and val datasets with their respective transforms
        train_set = TransformedSubset(full_dataset, train_indices, train_transform)
        val_set = TransformedSubset(full_dataset, val_indices, val_transform)
        test_set = TransformedSubset(full_dataset, val_indices, test_transform)

        img_dim = [3, 224, 224]
        num_classes = 397
    elif args.dataset == 'PET':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])


        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])


        train_set = datasets.OxfordIIITPet(
            root=data_dir, split='trainval', target_types='category',
            transform=train_transform, download=True
        )

        test_set = datasets.OxfordIIITPet(
            root=data_dir, split='test', target_types='category',
            transform=test_transform, download=True
        )

        val_set = datasets.OxfordIIITPet(
            root=data_dir, split='test', target_types='category',
            transform=val_transform, download=True
        )



        img_dim = [3, 224, 224]
        num_classes = 37
    elif args.dataset == 'DTD':

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # deterministic crop
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])


        # Load DTD with 'train', 'val', and 'test' splits
        train_set_2 = datasets.DTD(root=data_dir, split='train', transform=train_transform, download=True)
        train_set_1 = datasets.DTD(root=data_dir, split='val', transform=test_transform, download=True)

        train_set = ConcatDataset([train_set_1, train_set_2])

        test_set = datasets.DTD(root=data_dir, split='test', transform=test_transform, download=True)
        val_set = datasets.DTD(root=data_dir, split='test', transform=val_transform, download=True)
        img_dim = [3, 224, 224]
        num_classes = 47

    else:
        raise "Selected dataset '{}' not available.".format(args.dataset)

    # Dataloader
    train_loader = DataLoader(train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True
    )

    test_loader = DataLoader(test_set,
        batch_size=args.batch_size_test,
        num_workers=8,
        pin_memory=True,
        shuffle=False
    )

    if validation_split:
        val_loader = DataLoader(val_set,
            batch_size=args.batch_size_test,
            num_workers=8,
            pin_memory=True,
            shuffle=False
        )
    else:
        val_loader = test_loader

    print("train size", len(train_set))
    print("val size", len(val_set))
    print("test size", len(test_set))
    print("image 1 train:", type(train_set[0][0]))
    print("image 1 test:", type(test_set[0][0]))
    print("image 1 train:", train_set[0][0].shape)
    print("image 1 test:", test_set[0][0].shape)

    return train_loader, test_loader, val_loader, img_dim, num_classes
