import os
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
import matplotlib.pyplot as plt
import csv

logger = logging.getLogger(__name__)


def pre_transform_load(load_name: str, load_type: str, model: torch.nn.Module):
    """
    如果有指定 checkpoint，就加载它
    """
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
    1. 遍历 batch_size 和量化方式
    2. 执行 transform_graph
    3. 记录推理时间、准确率、推理能耗
    4. 不再存储任何模型权重或 checkpoint
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

    batch_sizes = config.get("batch_size", [64])  # 默认 batch_size = 64
    quant_methods = ["tensorrt_int8", "sparsity_int8", "tensorrt_fp16", "sparsity_fp16", "tensorrt_fp32"]

    # **确保 config["passes"] 是 `dict` 而不是 `list`**
    if isinstance(config["passes"], list):
        config["passes"] = {method: config.get(f"passes.{method}", {}) for method in quant_methods}

    # **检查是否所有 `quant_methods` 都存在于 `config["passes"]`**
    available_methods = list(config["passes"].keys())
    logger.info(f"Available quantization methods in config: {available_methods}")

    results_collector = []

    for bs in batch_sizes:
        for quant_method in quant_methods:
            logger.info(f"Running {model_name} | batch_size={bs} | quant={quant_method} ...")

            config["batch_size"] = bs

            # **检查 `quant_method` 是否存在于 `config["passes"]`**
            if quant_method not in config["passes"] or not config["passes"][quant_method]:
                logger.warning(f"Warning: {quant_method} not found in config['passes'], skipping...")
                continue  # **跳过当前量化方法，防止 KeyError**

            # **只使用当前 pass 进行实验**
            current_config = deepcopy(config)
            current_config["passes"] = {quant_method: config["passes"][quant_method]}

            # 生成 Forward Arguments
            cf_args = _prepare_cf_args(current_config, model, model_info, data_module, task, accelerator)

            # 执行 transform_graph（不存储模型）
            graph = transform_graph(
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
            )

            # 直接调用 _extract_results 并传入 cf_args
            metrics = _extract_results(graph, quant_method, cf_args)
            results_collector.append(metrics)

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
    如果你需要在 PyTorch module 级别做 transform，就在这里写逻辑。
    """
    accelerator = parse_accelerator(accelerator)
    model = pre_transform_load(load_name, load_type, model)
    model.to(accelerator)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pass_config = config["passes"]
    for pass_name, each_pass_cfg in pass_config.items():
        my_pass = MODULE_PASSES[pass_name]
        model, _ = my_pass(model, pass_args=each_pass_cfg)

    # 存储变换后的模型
    if save_dir is not None:
        ckpt_dir = save_dir / "transformed_ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state_dict_path = ckpt_dir / "state_dict.pt"
        torch.save(model.state_dict(), state_dict_path)
        logger.info(f"Module-level transform done; saved at {state_dict_path}")

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
):
    """
    1. 遍历 batch_size 和 quant_method
    2. 对每个组合执行 transformation
    3. 记录 (推理时间, 准确率, Inference Energy Consumption)
    4. ⚠️ 不再存储 transformed graph 或 checkpoint ⚠️
    """
    accelerator = parse_accelerator(accelerator)

    batch_sizes = config.get("batch_size", [64])
    if not isinstance(batch_sizes, list):
        batch_sizes = [batch_sizes]

    results_collector = []

    for bs in batch_sizes:
        logger.info(f"=== Start passes for batch_size={bs} ===")

        data_module.batch_size = bs
        data_module.prepare_data()
        data_module.setup()

        current_model = pre_transform_load(load_name, load_type, model)
        current_model.to(accelerator)

        # 生成 Forward Arguments
        cf_args = _prepare_cf_args(config, current_model, model_info, data_module, task, accelerator)

        # 构建 MaseGraph
        graph = MaseGraph(model=current_model, cf_args=cf_args)
        graph, _ = init_metadata_analysis_pass(graph, pass_args=None)

        if load_name is not None and load_type == "mz":
            graph, _ = load_mase_graph_interface_pass(graph, pass_args={"load_dir": load_name})
        else:
            dummy_in = get_dummy_input(model_info=model_info, data_module=data_module, task=task, device=accelerator)
            graph, _ = add_common_metadata_analysis_pass(graph, pass_args={"dummy_in": dummy_in})
            graph, _ = add_software_metadata_analysis_pass(graph, pass_args=None)

        passes_config = config["passes"]
        for pass_name, pass_cfg in passes_config.items():
            pass_cfg["batch_size"] = bs

        graph = _run_all_passes(graph, passes_config, data_module, task, model_info, accelerator, config, results_collector)

        logger.info(f"=== Finished passes for batch_size={bs} ===\n")

    #_plot_curves(results_collector, save_dir)

    logger.info("transform_graph complete.")
    return graph



def _extract_results(graph, quant_method, cf_args):
    """ 直接使用 cf_args 作为参数，而不是单独传递 batch_size, task, accelerator 等。 """
    # 确保 cf_args 包含 "dataset"
    if "dataset" not in cf_args:
        raise ValueError("cf_args 缺少 'dataset' 参数！")

    # 运行 runtime_analysis_pass，传入完整的 cf_args
    _, analysis_res = PASSES["runtime_analysis_pass"](graph, pass_args=cf_args)

    return {
        "quant_method": quant_method,
        "batch_size": cf_args.get("batch_size", 64),
        "accuracy": analysis_res.get("Average Accuracy", 0.0),
        "latency": analysis_res.get("Average Latency", 0.0),
        "energy": analysis_res.get("Inference Energy Consumption", 0.0),
    }



def _prepare_cf_args(config, model, model_info, data_module, task, accelerator):
    """
    如果没有 cf_args，就自动生成；如果有就用 config["cf_args"]。
    确保 cf_args 包含 batch_size, task, model, data_module, accelerator, dataset。
    """
    if "cf_args" in config:
        cf_args = config["cf_args"]
    else:
        cf_args = get_cf_args(model_info=model_info, task=task, model=model)

    # 确保 cf_args 包含所有必要的键
    if "batch_size" not in cf_args:
        cf_args["batch_size"] = config.get("batch_size", 64)
    if "task" not in cf_args:
        cf_args["task"] = task
    if "model" not in cf_args:
        cf_args["model"] = config.get("model")
    if "data_module" not in cf_args:
        cf_args["data_module"] = data_module
    if "accelerator" not in cf_args:
        cf_args["accelerator"] = accelerator.type
    if "dataset" not in cf_args:  # **添加 dataset 以避免 KeyError**
        cf_args["dataset"] = config.get("dataset", "default_dataset")  # **如果没有 dataset，默认一个值**

    return cf_args


def _run_all_passes(graph, passes_config, data_module, task, model_info, accelerator, config, results_collector):
    """
    遍历 pass，并在 'runtime_analysis_pass' 返回后把 (acc, lat) 存到 results_collector。
    """
    for pass_name, pass_cfg in passes_config.items():
        pass_cfg["task"] = task
        pass_cfg["dataset"] = config["dataset"]
        pass_cfg["model"] = config["model"]
        pass_cfg["data_module"] = data_module
        pass_cfg["accelerator"] = accelerator.type

        graph, results_collector = _dispatch_pass(pass_name, pass_cfg, graph, results_collector)
        assert isinstance(graph, MaseGraph), f"Pass {pass_name} did not return MaseGraph"
    return graph


def _dispatch_pass(pass_name, pass_cfg, graph, results_collector):
    """
    遍历 `transform_pass`，但不存储最终 graph
    """
    from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass

    match pass_name:
        case "runtime_analysis_ori":
            logger.info(">> runtime_analysis_ori pass ...")
            graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})

            ori_graph = deepcopy_mase_graph(graph)
            _, analysis_res = PASSES["runtime_analysis_pass"](ori_graph, pass_args=pass_cfg)

            if analysis_res is not None:
                results_collector.append({
                    "stage": "Original",
                    "accuracy": analysis_res.get("Average Accuracy", 0.0),
                    "latency": analysis_res.get("Average Latency", 0.0),
                    "batch_size": pass_cfg.get("batch_size", 1)
                })
            return graph, results_collector

        case "tensorrt_int8" | "tensorrt_fp16" | "tensorrt_fp32":
            logger.info(f">> {pass_name} pass ...")
            graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})

            graph, runtime_meta = PASSES["tensorrt_engine_interface_pass"](graph, pass_args=pass_cfg)
            _, analysis_res = PASSES["runtime_analysis_pass"](runtime_meta["trt_engine_path"], pass_args=pass_cfg)

            if analysis_res is not None:
                results_collector.append({
                    "stage": pass_name.upper(),
                    "accuracy": analysis_res.get("Average Accuracy", 0.0),
                    "latency": analysis_res.get("Average Latency", 0.0),
                    "batch_size": pass_cfg.get("batch_size", 1)
                })
            return graph, results_collector

        case "sparsity_fp16" | "sparsity_int8":
            logger.info(f">> {pass_name} pass ...")
            # 先对 graph 做 value cast
            graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})

            # 传递 precision 参数，FP16 还是 INT8
            precision = "fp16" if pass_name == "sparsity_fp16" else "int8"
            pass_cfg["precision"] = precision  # 让 `tensorrt_sparsity_interface_pass` 知道是 FP16 还是 INT8

            graph, runtime_meta = PASSES["tensorrt_sparsity_interface_pass"](graph, pass_args=pass_cfg)

            _, analysis_res = PASSES["runtime_analysis_pass"](runtime_meta["trt_engine_path"], pass_args=pass_cfg)
            if analysis_res is not None:
                acc = analysis_res.get("Average Accuracy", 0.0)
                lat = analysis_res.get("Average Latency", 0.0)
                stage = "Sparsity-FP16" if pass_name == "sparsity_fp16" else "Sparsity-INT8"
                bs = pass_cfg.get("batch_size", 1)
                results_collector.append({
                    "stage": stage,
                    "accuracy": acc,
                    "latency": lat,
                    "batch_size": bs
                })
            return graph, results_collector

        case _:
            logger.info(f">> {pass_name} pass (generic) ...")
            my_pass = PASSES[pass_name]
            graph, _ = my_pass(graph, pass_args=pass_cfg)
            return graph, results_collector


def _save_results(results_collector, save_dir, model_name):
    """保存实验结果到 CSV 文件"""
    if not isinstance(save_dir, Path):  # 确保 save_dir 是 Path 类型
        save_dir = Path(save_dir)

    # **确保目录存在，否则创建**
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = f"{model_name}_experiment_results.csv"
    csv_path = save_dir / csv_filename  # **现在不会报错**

    with open(csv_path, "w", newline="") as f:
        fieldnames = ["model_name", "quant_method", "batch_size", "latency", "accuracy", "energy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in results_collector:
            d["model_name"] = model_name
            writer.writerow(d)

    logger.info(f"Results saved to: {csv_path}")



def _plot_curves(results_collector, save_dir, model_name):
    """绘制实验曲线：不同 batch_size 的 (latency, accuracy) 关系，并在文件名中加入 model_name"""

    if not isinstance(save_dir, Path):  # **确保 `save_dir` 是 Path 类型**
        save_dir = Path(save_dir)
    
    plt.figure(figsize=(8, 6))

    colors = {
        "tensorrt_int8": "red",
        "tensorrt_fp16": "blue",
        "sparsity_fp16": "green",
        "sparsity_int8": "purple", 
        "tensorrt_fp32": "orange",
    }

    for quant_method in colors.keys():
        data = [d for d in results_collector if d["quant_method"] == quant_method]
        if not data:
            continue

        latencies = [d["latency"] for d in data]
        accuracies = [d["accuracy"] for d in data]

        plt.plot(latencies, accuracies, marker="o", color=colors[quant_method], label=f"{quant_method}")

    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title(f"Latency vs Accuracy for {model_name}")
    plt.legend()

    plot_filename = f"{model_name}_experiment_plot.png"
    plot_path = save_dir / plot_filename  # **确保路径不会报错**

    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Plot saved to: {plot_path}")

