# -*- coding: utf-8 -*-

import argparse
import pickle as pkl
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed
from ase.units import GPa
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_ema import ExponentialMovingAverage

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.m3gnet.scaling import AtomScaling
from mattersim.forcefield.potential import Potential
from mattersim.utils.atoms_utils import AtomsAdaptor


# ============================================================
# 1. 基础函数
# ============================================================
def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_atoms_dataset(data_path):
    """
    Load structures from a pkl file or another structure file supported
    by AtomsAdaptor.
    """
    if data_path is None:
        return None

    data_path = str(data_path)

    if data_path.endswith(".pkl"):
        with open(data_path, "rb") as file:
            atoms_list = pkl.load(file)
    else:
        atoms_list = AtomsAdaptor.from_file(filename=data_path)

    return atoms_list


def extract_labels(atoms_list, include_forces=True, include_stresses=True):
    """
    Extract energies, forces, and stresses from ASE Atoms objects.
    """
    energies = []
    forces = [] if include_forces else None
    stresses = [] if include_stresses else None

    for atoms in atoms_list:
        energies.append(atoms.get_potential_energy())

        if include_forces:
            forces.append(atoms.get_forces())

        if include_stresses:
            # ASE stress is converted to GPa.
            stresses.append(
                atoms.get_stress(voigt=False) / GPa
            )

    return energies, forces, stresses


def build_dataset_dataloader(atoms_list, args, shuffle):
    """
    Construct a MatterSim dataloader.
    """
    energies, forces, stresses = extract_labels(
        atoms_list=atoms_list,
        include_forces=args.include_forces,
        include_stresses=args.include_stresses,
    )

    dataloader = build_dataloader(
        atoms_list,
        energies,
        forces,
        stresses,
        shuffle=shuffle,
        pin_memory=True,
        **vars(args),
    )

    return dataloader


def reinitialize_prediction_head(model):
    """
    Reinitialize the MatterSim M3GNet prediction head.

    In MatterSim-v1.0.0 M3GNet, model.final is the final GatedMLP
    that maps atomic features to atomic energies.
    """
    if not hasattr(model, "final"):
        raise AttributeError(
            "The loaded MatterSim model does not contain `model.final`. "
            "Please check the installed MatterSim version."
        )

    prediction_head = model.final

    for module in prediction_head.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    return prediction_head


def configure_optimizer(potential, args):
    """
    Configure separate learning rates for the backbone and prediction head.
    """
    model = potential.model

    # All parameters remain trainable.
    for parameter in model.parameters():
        parameter.requires_grad = True

    prediction_head = model.final

    head_parameters = list(prediction_head.parameters())
    head_parameter_ids = {
        id(parameter) for parameter in head_parameters
    }

    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in head_parameter_ids
    ]

    if len(head_parameters) == 0:
        raise RuntimeError(
            "No trainable parameters were found in the prediction head."
        )

    if len(backbone_parameters) == 0:
        raise RuntimeError(
            "No trainable parameters were found in the backbone."
        )

    optimizer = Adam(
        [
            {
                "params": backbone_parameters,
                "lr": args.backbone_lr,
            },
            {
                "params": head_parameters,
                "lr": args.head_lr,
            },
        ],
        eps=1e-7,
    )

    scheduler = StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    # The head has been reinitialized, so EMA must also be recreated.
    ema = ExponentialMovingAverage(
        model.parameters(),
        decay=args.ema_decay,
    )

    potential.optimizer = optimizer
    potential.scheduler = scheduler
    potential.ema = ema

    backbone_parameter_count = sum(
        parameter.numel()
        for parameter in backbone_parameters
        if parameter.requires_grad
    )

    head_parameter_count = sum(
        parameter.numel()
        for parameter in head_parameters
        if parameter.requires_grad
    )

    print("\nModel parameter groups")
    print("-" * 60)
    print(
        f"Backbone parameters       : "
        f"{backbone_parameter_count:,}"
    )
    print(
        f"Prediction-head parameters: "
        f"{head_parameter_count:,}"
    )
    print(
        f"Backbone learning rate    : "
        f"{args.backbone_lr:.2e}"
    )
    print(
        f"Prediction-head lr        : "
        f"{args.head_lr:.2e}"
    )
    print("-" * 60)

    return potential


# ============================================================
# 2. 主程序
# ============================================================
def main(args):
    set_random_seed(args.seed)

    device = (
        "cuda"
        if torch.cuda.is_available() and args.device == "cuda"
        else "cpu"
    )

    args.device = device
    args_dict = vars(args)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Running model on: {device}")

    # ========================================================
    # 2.1 读取训练集
    # ========================================================
    print("\nLoading training dataset...")

    atoms_train = load_atoms_dataset(
        args.train_data_path
    )

    if atoms_train is None or len(atoms_train) == 0:
        raise ValueError("The training dataset is empty.")

    print(
        f"Number of training configurations: "
        f"{len(atoms_train)}"
    )

    train_dataloader = build_dataset_dataloader(
        atoms_list=atoms_train,
        args=args,
        shuffle=True,
    )

    # ========================================================
    # 2.2 读取验证集
    # ========================================================
    if args.valid_data_path is not None:
        print("\nLoading validation dataset...")

        atoms_val = load_atoms_dataset(
            args.valid_data_path
        )

        if len(atoms_val) == 0:
            raise ValueError("The validation dataset is empty.")

        print(
            f"Number of validation configurations: "
            f"{len(atoms_val)}"
        )

        val_dataloader = build_dataset_dataloader(
            atoms_list=atoms_val,
            args=args,
            shuffle=False,
        )
    else:
        atoms_val = None
        val_dataloader = None

    # ========================================================
    # 2.3 读取测试集
    # ========================================================
    if args.test_data_path is not None:
        print("\nLoading held-out test dataset...")

        atoms_test = load_atoms_dataset(
            args.test_data_path
        )

        if len(atoms_test) == 0:
            raise ValueError("The test dataset is empty.")

        print(
            f"Number of held-out test configurations: "
            f"{len(atoms_test)}"
        )

        test_dataloader = build_dataset_dataloader(
            atoms_list=atoms_test,
            args=args,
            shuffle=False,
        )
    else:
        atoms_test = None
        test_dataloader = None

    # ========================================================
    # 2.4 输出数据集划分信息
    # ========================================================
    n_train = len(atoms_train)
    n_val = len(atoms_val) if atoms_val is not None else 0
    n_test = len(atoms_test) if atoms_test is not None else 0
    n_total = n_train + n_val + n_test

    print("\nDataset summary")
    print("-" * 60)
    print(f"Training set  : {n_train}")
    print(f"Validation set: {n_val}")
    print(f"Test set      : {n_test}")
    print(f"Total         : {n_total}")

    if n_total > 0:
        print(
            "Split ratio   : "
            f"{100 * n_train / n_total:.2f}% / "
            f"{100 * n_val / n_total:.2f}% / "
            f"{100 * n_test / n_total:.2f}%"
        )

    print("-" * 60)

    # ========================================================
    # 2.5 可选的 energy normalization
    # ========================================================
    if args.re_normalize:
        print("\nConstructing energy normalization module...")

        train_energies, train_forces, _ = extract_labels(
            atoms_list=atoms_train,
            include_forces=args.include_forces,
            include_stresses=args.include_stresses,
        )

        scale = AtomScaling(
            atoms=atoms_train,
            total_energy=train_energies,
            forces=train_forces,
            verbose=True,
            **args_dict,
        ).to(device)
    else:
        scale = None

    # ========================================================
    # 2.6 加载预训练 MatterSim
    # ========================================================
    print("\nLoading pretrained MatterSim model...")

    potential = Potential.from_checkpoint(
        load_path=args.load_model_path,
        load_training_state=False,
        **args_dict,
    )

    potential.model = potential.model.to(device)

    if scale is not None:
        potential.model.set_normalizer(scale)

    # ========================================================
    # 2.7 重新初始化 prediction head
    # ========================================================
    print("\nReinitializing prediction head...")

    reinitialize_prediction_head(
        potential.model
    )

    # ========================================================
    # 2.8 设置分组学习率
    # ========================================================
    potential = configure_optimizer(
        potential=potential,
        args=args,
    )

    # ========================================================
    # 2.9 Fine-tuning
    # ========================================================
    print("\nStarting MatterSim fine-tuning...")

    potential.train_model(
        train_dataloader,
        val_dataloader,
        loss=torch.nn.HuberLoss(
            delta=args.huber_delta
        ),
        metric_name="val_loss",
        **args_dict,
    )

    print("\nFine-tuning completed.")

    # ========================================================
    # 2.10 使用 held-out test set 评价 best model
    # ========================================================
    if test_dataloader is not None:
        best_model_path = (
            save_path / "best_model.pth"
        )

        if not best_model_path.exists():
            raise FileNotFoundError(
                f"Best model was not found: "
                f"{best_model_path}"
            )

        print(
            "\nLoading the best model for held-out "
            "test-set evaluation..."
        )

        best_potential = Potential.from_checkpoint(
            load_path=str(best_model_path),
            load_training_state=False,
            **args_dict,
        )

        best_potential.model = (
            best_potential.model.to(device)
        )

        test_metrics = best_potential.test_model(
            test_dataloader,
            loss=torch.nn.HuberLoss(
                delta=args.huber_delta
            ),
            include_energy=True,
            include_forces=args.include_forces,
            include_stresses=args.include_stresses,
            wandb=False,
        )

        test_loss = test_metrics[0]
        test_energy_mae = test_metrics[1]
        test_force_mae = test_metrics[2]
        test_stress_mae = test_metrics[3]

        print("\nHeld-out test-set results")
        print("-" * 60)
        print(
            f"Test loss       : "
            f"{test_loss:.8f}"
        )
        print(
            f"Energy MAE      : "
            f"{test_energy_mae:.8f} eV/atom"
        )
        print(
            f"Force MAE       : "
            f"{test_force_mae:.8f} eV/Å"
        )
        print(
            f"Stress MAE      : "
            f"{test_stress_mae:.8f} GPa"
        )
        print("-" * 60)


# ============================================================
# 3. 参数设置
# ============================================================
args = argparse.Namespace(
    # --------------------------------------------------------
    # 文件路径
    # --------------------------------------------------------
    train_data_path="/PATH/TO/TRAIN.pkl",
    valid_data_path="/PATH/TO/VAL.pkl",
    test_data_path="/PATH/TO/TEST.pkl",

    load_model_path=(
        "/root/shared-nvme/suchen/software/"
        "mattersim-v1.0.0-5M.pth"
    ),

    save_path="/PATH/TO/SAVE",

    # --------------------------------------------------------
    # 运行设置
    # --------------------------------------------------------
    run_name="alloy",
    device="cuda",
    seed=42,

    # --------------------------------------------------------
    # 模型参数
    # --------------------------------------------------------
    cutoff=5.0,
    threebody_cutoff=4.0,

    # --------------------------------------------------------
    # 训练参数
    # --------------------------------------------------------
    epochs=200,
    batch_size=4,

    # 这里的 lr 仅用于兼容 MatterSim 的模型加载接口。
    # 实际训练使用下面两个分组学习率。
    lr=1e-4,

    backbone_lr=1e-4,
    head_lr=2e-3,

    step_size=10,
    gamma=0.95,
    ema_decay=0.99,

    huber_delta=0.01,

    # --------------------------------------------------------
    # 损失函数设置
    # --------------------------------------------------------
    include_forces=True,
    include_stresses=True,

    force_loss_ratio=1.0,
    stress_loss_ratio=0.1,

    # --------------------------------------------------------
    # Early stopping 和模型保存
    # --------------------------------------------------------
    early_stop_patience=10,

    save_checkpoint=True,
    ckpt_interval=10,

    # --------------------------------------------------------
    # Energy normalization
    # --------------------------------------------------------
    re_normalize=False,

    # --------------------------------------------------------
    # WandB
    # --------------------------------------------------------
    wandb=False,
)


main(args)