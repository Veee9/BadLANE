import argparse
import ast
import numpy
import torch
import torchvision
import data.datasets as datasets
from utils.tools import count_parameters
from models.cglow import CondGlowModel as Generator
from trainners import MetaLearner, Learner
from utils.load_models import load_laneatt_model, load_ufld_model, load_polylanenet_model, load_resa_model
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train c-Glow')

    # input output path
    parser.add_argument("-d", "--dataset_name", type=str)
    parser.add_argument("--train_dataset_root", type=str)
    parser.add_argument("--valid_dataset_root", type=str)

    # log root
    parser.add_argument("--log_root", type=str, default="")

    # C-Glow parameters
    parser.add_argument("--x_size", type=tuple, default=(3, 224, 224))
    parser.add_argument("--y_size", type=tuple, default=(3, 224, 224))
    parser.add_argument("--x_hidden_channels", type=int, default=128)
    parser.add_argument("--x_hidden_size", type=int, default=64)
    parser.add_argument("--y_hidden_channels", type=int, default=256)
    parser.add_argument("-K", "--flow_depth", type=int, default=8)
    parser.add_argument("-L", "--num_levels", type=int, default=3)
    parser.add_argument("--learn_top", type=ast.literal_eval, default=False)

    # Dataset preprocess parameters
    parser.add_argument("--label_scale", type=float, default=1)
    parser.add_argument("--label_bias", type=float, default=0.0)
    parser.add_argument("--x_bins", type=float, default=256.0)
    parser.add_argument("--y_bins", type=float, default=2.0)

    # Optimizer parameters
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.9999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--regularizer", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--Lambda", type=float, default=5e-1)

    # Trainer parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--down_sample_x", type=int, default=8)
    parser.add_argument("--down_sample_y", type=int, default=8)
    parser.add_argument("--max_grad_clip", type=float, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--test_gap", type=int, default=1000)
    parser.add_argument("--log_gap", type=int, default=1)
    parser.add_argument("--inference_gap", type=int, default=1000)
    parser.add_argument("--save_gap", type=int, default=1000)
    parser.add_argument("--adv_loss", type=ast.literal_eval, default=False)
    parser.add_argument("--tanh", type=ast.literal_eval, default=False)
    parser.add_argument("--only", type=ast.literal_eval, default=False)
    parser.add_argument("--clamp", type=ast.literal_eval, default=False)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--class_size", type=int, default=-1)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--linf", type=float, default=0.1)

    # Adv augmentation
    parser.add_argument("--adv_aug", type=ast.literal_eval, default=False)
    parser.add_argument("--adv_rand", type=ast.literal_eval, default=False)
    parser.add_argument("--nes", type=ast.literal_eval, default=False)
    parser.add_argument("--new_form", type=ast.literal_eval, default=False)
    parser.add_argument("--normalize_grad", type=ast.literal_eval, default=False)
    parser.add_argument("--adv_epoch", type=int, default=0)

    # model path
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--name", type=str, default="")

    # LD backbone
    parser.add_argument("--config_path", type=str, default='none')
    parser.add_argument("--checkpoint_path", type=str, default='none')

    # CMA args
    parser.add_argument("--cma_update_max_epoch", type=int, default=20)
    parser.add_argument("--cma_popsize", type=int, default=8)
    parser.add_argument("--cma_sigma0", type=int, default=10000)
    parser.add_argument("--cma_seed", type=int, default=666)
    parser.add_argument("--cma_ftarget", type=int, default=0)  # TODO not sure the ftarget is correct

    # Meta args
    parser.add_argument("--support_set", type=str, default="Resnet18")
    parser.add_argument("--query_set", type=str, default="Resnet18")
    parser.add_argument("--curriculum", type=ast.literal_eval, default=False, help="use only hard data")
    parser.add_argument("--meta_iteration", type=int, default=1)

    # Target attack args
    parser.add_argument("--target", type=ast.literal_eval, default=False)
    parser.add_argument("--target_label", type=int, default=None)
    # Open-set
    parser.add_argument("--openset", type=ast.literal_eval, default=False)

    args = parser.parse_args()
    cuda = torch.cuda.is_available()

    if args.dataset_name == 'tusimple' or args.dataset_name == 'culane':

        train_set = datasets.AdvTrainDataset(root_dir=args.train_dataset_root)
        valid_set = datasets.AdvTrainDataset(root_dir=args.valid_dataset_root)

        # train_set = datasets.AdvTrainDatasetFile(root_dir=args.train_dataset_root)
        # valid_set = datasets.AdvTrainDatasetFile(root_dir=args.valid_dataset_root)
        args.x_size, args.y_size = (3, 100, 100), (3, 100, 100)
    else:
        raise NotImplementedError

    generator = Generator(args).cuda()
    print('Data saved file in: ', args.log_root, args.name)
    print("number of param: {}".format(count_parameters(generator)))

    # optimizer
    optim = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.regularizer)

    if args.model_path != "":
        print('Loading pre-trained model from path: ', args.model_path)
        state = torch.load(args.model_path, map_location='cuda')
        optim.load_state_dict(state["optim"])
        generator.load_state_dict(state["model"])
        del state

    if args.adv_loss:
        support_set_models, query_set_models = [], []
        if args.dataset_name == 'tusimple':
            if args.support_set == 'laneatt':
                load_model = load_laneatt_model
            elif args.support_set == 'ufld':
                load_model = load_ufld_model
            elif args.support_set == 'polylanenet':
                load_model = load_polylanenet_model
            elif args.support_set == 'resa':
                load_model = load_resa_model

            for model_name in args.support_set.split(","):
                support_set_models.append(load_model(args.config_path, args.checkpoint_path))
        elif args.dataset_name == 'culane':
            if args.support_set == 'laneatt':
                model = load_laneatt_model(args.config_path, args.checkpoint_path).cuda()
            support_set_models.append(model)
        # for model_name in args.query_set.split(","):
        #     query_set_models.append(load_model(model_name))

        trainer = MetaLearner.Trainer(generator, support_set_models, query_set_models, optim, train_set, valid_set, args, cuda)
    else:
        trainer = Learner.Trainer(generator, optim, train_set, valid_set, args, cuda)

    trainer.train()

