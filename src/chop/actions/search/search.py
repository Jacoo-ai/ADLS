import logging
from os import PathLike

import toml
import torch

from ...tools.checkpoint_load import load_model
from ...tools.config_load import load_config
from ...tools.get_input import get_dummy_input
from .search_space import get_search_space_cls
from .strategies import get_search_strategy_cls
from chop.tools.utils import device
from chop.tools.utils import parse_accelerator

logger = logging.getLogger(__name__)


def parse_search_config(
    search_config: dict,
):
    """
    Parse search config from a dict or a toml file and do sanity check. The search config must consist of two parts: strategy and search_space.

    Args:
        search_config: A dictionary or a path to a toml file containing the search config.

    Returns:
        _type_: _description_
    """
    if not isinstance(search_config, dict):
        search_config = load_config(search_config)
    search_config = search_config["search"]  # the actual config for action search
    strategy_config = search_config["strategy"]
    search_space_config = search_config["search_space"]

    return strategy_config, search_space_config


def search(
    model: torch.nn.Module,
    model_info,
    task: str,
    dataset_info,
    data_module,
    search_config: dict | PathLike,
    save_path: PathLike,
    accelerator: str,
    load_name: PathLike = None,
    load_type: str = None,
    visualizer=None,
):
    """
    Perform search using a defined search strategy and a search space.

    Args:
        model (torch.nn.Module): _description_
        model_info (_type_): _description_
        task (str): _description_
        dataset_info (_type_): _description_
        data_module (_type_): _description_
        search_config (dict | PathLike): _description_
        save_path (PathLike): _description_
        accelerator (str): _description_
        load_name (PathLike, optional): _description_. Defaults to None.
        load_type (str, optional): _description_. Defaults to None.
        visualizer (_type_, optional): _description_. Defaults to None.
    """

    # search preparation
    accelerator = parse_accelerator(accelerator)
    strategy_config, search_space_config = parse_search_config(search_config)
    save_path.mkdir(parents=True, exist_ok=True)

    # load model if the save_name is provided
    if load_name is not None and load_type in ["pl", "mz", "pt"]:
        model = load_model(load_name=load_name, load_type=load_type, model=model)
        logger.info(f"Loaded model from {load_name}.")
    model.to(accelerator)
    # set up data module
    data_module.prepare_data()
    data_module.setup()

    # construct the search space
    logger.info("Building search space...")
    search_space_cls = get_search_space_cls(search_space_config["name"])
    search_space = search_space_cls(
        model=model,
        model_info=model_info,
        config=search_space_config,
        dummy_input=get_dummy_input(model_info, data_module, task, device=accelerator),
        accelerator=accelerator,
        data_module=data_module,
    )
    search_space.build_search_space()

    # construct a search strategy
    strategy_cls = get_search_strategy_cls(strategy_config["name"])
    strategy = strategy_cls(
        model_info=model_info,
        task=task,
        dataset_info=dataset_info,
        data_module=data_module,
        config=strategy_config,
        accelerator=accelerator,
        save_dir=save_path,
        visualizer=visualizer,
    )

    logger.info("Search started...")
    # perform search and save the results
    strategy.search(search_space)
