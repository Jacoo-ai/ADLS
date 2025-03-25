import os
import csv
import logging
from copy import deepcopy
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph import PASSES
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    init_metadata_analysis_pass,
)
from chop.passes.graph.interface import (
    load_mase_graph_interface_pass,
    save_mase_graph_interface_pass,
)
from chop.passes.graph.utils import deepcopy_mase_graph
from chop.tools.checkpoint_load import load_model
from chop.tools.config_load import load_config
from chop.tools.get_input import get_cf_args, get_dummy_input
from chop.tools.utils import parse_accelerator, to_numpy_if_tensor
from chop.passes.module import PASSES as MODULE_PASSES
from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass

logger = logging.getLogger(__name__)



def pre_transform_load(load_name: str, load_type: str, model: torch.nn.Module):
    if load_name is not None and load_type in ["pt", "pl", "mz"]:
        if not os.path.exists(load_name):
            raise ValueError(f"file or directory not found: {load_name}")
        model = load_model(load_name=load_name, load_type=load_type, model=model)
    elif load_name is not None and load_type == "hf":
        model = load_model(load_name=load_name, load_type=load_type, model=model)
    return model


def transmeta(
    model: torch.nn.Module,
    model_info: dict,
    model_name: str,
    data_module,
    task: str,
    config: str,
    save_dir: str = None,
    load_name: str = None,
    load_type: str = None,
    accelerator: str = "auto",
):
    """
    Run transformation passes across batch sizes and quantization methods.
    Records latency, accuracy, and energy for each configuration.
    Does NOT save model weights or checkpoints.
    """
    accelerator = parse_accelerator(accelerator)
    model = pre_transform_load(load_name, load_type, model)
    model.to(accelerator)

    config = load_config(config)
    model_name = config.get("model", None)
    task = config.get("task", None)

    if model_name is None:
        raise KeyError("Config is missing 'model' key!")
    if task is None:
        raise KeyError("Config is missing 'task' key!")

    batch_sizes = config.get("batch_size", [64])
    quant_methods = ["tensorrt_int8", "sparsity_int8", "tensorrt_fp16", "sparsity_fp16", "tensorrt_fp32"]

    # Ensure passes are in dict format
    if isinstance(config["passes"], list):
        config["passes"] = {method: config.get(f"passes.{method}", {}) for method in quant_methods}

    logger.info(f"Available quantization methods in config: {list(config['passes'].keys())}")
    results_collector = []

    for bs in batch_sizes:
        for quant_method in quant_methods:
            logger.info(f"Running {model_name} | batch_size={bs} | quant={quant_method} ...")
            config["batch_size"] = bs

            if quant_method not in config["passes"] or not config["passes"][quant_method]:
                logger.warning(f"Skipping {quant_method} — not found in config['passes']")
                continue

            current_config = deepcopy(config)
            current_config["passes"] = {quant_method: config["passes"][quant_method]}

            cf_args = _prepare_cf_args(current_config, model, model_info, data_module, task, accelerator)

            transform_graph(
                model=model,
                model_info=model_info,
                model_name=model_name,
                data_module=data_module,
                task=task,
                config=current_config,
                save_dir=None,
                load_name=load_name,
                load_type=load_type,
                accelerator=accelerator,
                results_collector=results_collector,
            )

    _save_results(results_collector, save_dir, model_name)
    _plot_curves(results_collector, save_dir, model_name)
    logger.info("All experiments completed successfully.")



def transform_module(
    model: torch.nn.Module,
    model_info,
    model_name,
    data_module,
    task: str,
    config: dict,
    save_dir: str = None,
    load_name: str = None,
    load_type: str = None,
    accelerator: str = "auto",
):
    """
    Apply module-level transformation passes on a PyTorch model.
    Saves the state_dict after transformation.
    """
    accelerator = parse_accelerator(accelerator)
    model = pre_transform_load(load_name, load_type, model)
    model.to(accelerator)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for pass_name, each_pass_cfg in config["passes"].items():
        model, _ = MODULE_PASSES[pass_name](model, pass_args=each_pass_cfg)

    if save_dir:
        ckpt_dir = save_dir / "transformed_ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "state_dict.pt")
        logger.info(f"Module-level transform saved at: {ckpt_dir / 'state_dict.pt'}")

    return model



def transform_graph(
    model: torch.nn.Module,
    model_info,
    model_name,
    data_module,
    task: str,
    config: dict,
    save_dir: str = None,
    load_name: str = None,
    load_type: str = None,
    accelerator: str = "auto",
    results_collector: list = None,
):
    """
    Build MaseGraph, run transformation passes, and collect performance metrics.
    Does not persist transformed graphs.
    """
    accelerator = parse_accelerator(accelerator)
    batch_sizes = config.get("batch_size", [64])
    batch_sizes = [batch_sizes] if not isinstance(batch_sizes, list) else batch_sizes

    results_collector = results_collector or []

    for bs in batch_sizes:
        logger.info(f"=== Start passes for batch_size={bs} ===")

        data_module.batch_size = bs
        data_module.prepare_data()
        data_module.setup()

        current_model = pre_transform_load(load_name, load_type, model)
        current_model.to(accelerator)

        cf_args = _prepare_cf_args(config, current_model, model_info, data_module, task, accelerator)

        graph = MaseGraph(model=current_model, cf_args=cf_args)
        graph, _ = init_metadata_analysis_pass(graph, pass_args=None)

        if load_name and load_type == "mz":
            graph, _ = load_mase_graph_interface_pass(graph, pass_args={"load_dir": load_name})
        else:
            dummy_in = get_dummy_input(model_info=model_info, data_module=data_module, task=task, device=accelerator)
            graph, _ = add_common_metadata_analysis_pass(graph, pass_args={"dummy_in": dummy_in})
            graph, _ = add_software_metadata_analysis_pass(graph, pass_args=None)

        for pass_name, pass_cfg in config["passes"].items():
            pass_cfg["batch_size"] = bs

        graph = _run_all_passes(graph, config["passes"], data_module, task, model_info, accelerator, config, results_collector)

        logger.info(f"=== Finished passes for batch_size={bs} ===")

    logger.info("transform_graph complete.")
    return graph



def _prepare_cf_args(config, model, model_info, data_module, task, accelerator):
    """
    Generate or fetch cf_args (forward context) and ensure required keys are set.
    """
    cf_args = config.get("cf_args", get_cf_args(model_info=model_info, task=task, model=model))

    cf_args.setdefault("batch_size", config.get("batch_size", 64))
    cf_args.setdefault("task", task)
    cf_args.setdefault("model", config.get("model"))
    cf_args.setdefault("data_module", data_module)
    cf_args.setdefault("accelerator", accelerator.type)
    cf_args.setdefault("dataset", config.get("dataset", "default_dataset"))

    return cf_args


def _run_all_passes(graph, passes_config, data_module, task, model_info, accelerator, config, results_collector):
    """
    Execute each pass and collect runtime metrics when applicable.
    """
    for pass_name, pass_cfg in passes_config.items():
        pass_cfg.update({
            "task": task,
            "dataset": config["dataset"],
            "model": config["model"],
            "data_module": data_module,
            "accelerator": accelerator.type,
        })

        graph, results_collector = _dispatch_pass(pass_name, pass_cfg, graph, results_collector)
        assert isinstance(graph, MaseGraph), f"Pass {pass_name} did not return MaseGraph"
    return graph


def _dispatch_pass(pass_name, pass_cfg, graph, results_collector):
    """
    Dispatch and run a single transformation pass. Collect metrics if runtime analysis is triggered.
    """
    logger.info(f">> {pass_name} pass ...")
    graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})

    if pass_name in ["tensorrt_int8", "tensorrt_fp16", "tensorrt_fp32"]:
        graph, runtime_meta = PASSES["tensorrt_engine_interface_pass"](graph, pass_args=pass_cfg)
        _, analysis_res = PASSES["runtime_analysis_pass"](runtime_meta["trt_engine_path"], pass_args=pass_cfg)

    elif pass_name in ["sparsity_fp16", "sparsity_int8"]:
        pass_cfg["precision"] = "fp16" if "fp16" in pass_name else "int8"
        graph, runtime_meta = PASSES["tensorrt_sparsity_interface_pass"](graph, pass_args=pass_cfg)
        _, analysis_res = PASSES["runtime_analysis_pass"](runtime_meta["trt_engine_path"], pass_args=pass_cfg)

    else:
        graph, _ = PASSES[pass_name](graph, pass_args=pass_cfg)
        return graph, results_collector  # No metrics expected

    if analysis_res:
        results_collector.append({
            "quant_method": pass_name,
            "batch_size": pass_cfg.get("batch_size", 64),
            "accuracy": analysis_res.get("Average Accuracy", 0.0),
            "latency": analysis_res.get("Average Latency", 0.0),
            "energy": analysis_res.get("Inference Energy Consumption", 0.0)
        })

    return graph, results_collector



def _save_results(results_collector, save_dir, model_name):
    """
    Save results to CSV.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / f"{model_name}_experiment_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "quant_method", "batch_size", "latency", "accuracy", "energy"])
        writer.writeheader()
        for d in results_collector:
            d["model_name"] = model_name
            writer.writerow(d)

    logger.info(f"Results saved to: {csv_path}")


def _plot_curves(results_collector, save_dir, model_name):
    """
    Plot latency vs. accuracy for each quantization method.
    """
    save_dir = Path(save_dir)
    plt.figure(figsize=(8, 6))

    colors = {
        "tensorrt_int8": "red",
        "tensorrt_fp16": "blue",
        "sparsity_fp16": "green",
        "sparsity_int8": "purple",
        "tensorrt_fp32": "orange",
    }

    for quant_method, color in colors.items():
        data = [d for d in results_collector if d["quant_method"] == quant_method]
        if not data:
            continue
        latencies = [d["latency"] for d in data]
        accuracies = [d["accuracy"] for d in data]
        plt.plot(latencies, accuracies, marker="o", color=color, label=quant_method)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title(f"Latency vs Accuracy for {model_name}")
    plt.legend()

    plot_path = save_dir / f"{model_name}_experiment_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Plot saved to: {plot_path}")
