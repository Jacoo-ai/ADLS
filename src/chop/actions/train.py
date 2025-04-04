import logging
import os
from pathlib import Path

import pytorch_lightning as pl
from chop.tools.plt_wrapper import get_model_wrapper
from chop.tools.checkpoint_load import load_model
from chop.tools.get_input import get_dummy_input
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.passes.graph.interface import save_mase_graph_interface_pass
from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
from chop.ir.graph import MaseGraph
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from torch.distributed.fsdp import FullyShardedDataParallel
from pytorch_lightning.strategies import DDPStrategy
from chop.tools.utils import parse_accelerator, to_numpy_if_tensor


logger = logging.getLogger(__name__)


def train(
    model: pl.LightningModule,
    model_info: dict,
    data_module: pl.LightningDataModule,
    dataset_info: dict,
    task: str,
    optimizer: str,
    learning_rate: float,
    weight_decay: float,
    scheduler_args: dict,
    plt_trainer_args: dict,
    auto_requeue: bool,
    save_path: str,
    visualizer: TensorBoardLogger,
    load_name: str,
    load_type: str,
):
    """
    Train the model using PyTorch Lightning.

    Args:
        model (pl.LightningModule): Model to be trained.
        model_info (dict): Information about the model.
        data_module (pl.LightningDataModule): Data module for the model.
        dataset_info (dict): Information about the dataset.
        task (str): Task to be performed.
        optimizer (str): Optimizer to be used.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        scheduler_args (dict): Arguments for the scheduler.
        plt_trainer_args (dict): Arguments for PyTorch Lightning Trainer.
        auto_requeue (bool): Requeue on SLURM.
        save_path (str): Path to save the model.
        visualizer (TensorBoardLogger): Tensorboard logger.
        load_name (str): Name of the checkpoint to load.
        load_type (str): Type of the checkpoint to load.
    """
    if save_path is not None:
        # if save_path is None, the model will not be saved
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss_epoch",
            mode="min",
            filename="best",
            dirpath=save_path,
            save_last=True,
        )
        # tb_logger = TensorBoardLogger(save_dir=save_path, name="logs")
        lr_monitor_callback = LearningRateMonitor(logging_interval="step")
        plt_trainer_args["callbacks"] = [
            checkpoint_callback,
            lr_monitor_callback,
        ]
        plt_trainer_args["logger"] = visualizer

    # plugin
    if auto_requeue:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    plt_trainer_args["plugins"] = plugins

    wrapper_cls = get_model_wrapper(model_info, task)

    if load_name is not None:
        model = load_model(load_name, load_type=load_type, model=model)
        logger.info(f"'{load_type}' checkpoint loaded before training")

    pl_model = wrapper_cls(
        model,
        dataset_info=dataset_info,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_args=scheduler_args,
        epochs=plt_trainer_args["max_epochs"],
        optimizer=optimizer,
    )

    trainer = pl.Trainer(**plt_trainer_args)

    trainer.fit(
        pl_model,
        datamodule=data_module,
    )

    # Save the trained model along with relevant metadata in the training_ckpts folder.
    # NOTE: This is important if the model was previously transformed with architectural
    # changes. The state dictionary that's saved by PyTorch Lightning wouldn't work.
    if save_path is not None and load_name is not None and load_type == "mz":
        accelerator = plt_trainer_args["accelerator"]
        accelerator = parse_accelerator(accelerator)
        graph = MaseGraph(model)
        dummy_input = get_dummy_input(model_info, data_module, task, device=accelerator)
        graph, _ = init_metadata_analysis_pass(graph, None)
        graph, _ = add_common_metadata_analysis_pass(graph, {"dummy_in": dummy_input})
        graph, _ = add_software_metadata_analysis_pass(graph, None)
        transformed_ckpt = Path(save_path) / "transformed_ckpt"
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
        graph, _ = metadata_value_type_cast_transform_pass(
            graph, pass_args={"fn": to_numpy_if_tensor}
        )
        save_mase_graph_interface_pass(graph, pass_args=transformed_ckpt)
