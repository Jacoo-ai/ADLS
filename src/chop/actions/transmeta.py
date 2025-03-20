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
    1. 逐个加载 batch_size 和 量化方式
    2. 执行 transform_graph
    3. 记录 (推理时间, 准确率, Inference Energy Consumption)
    4. ⚠️ 不再存储任何模型权重或 checkpoint ⚠️
    """
    accelerator = parse_accelerator(accelerator)
    model = pre_transform_load(load_name, load_type, model)
    model.to(accelerator)

    config = load_config(config)
    batch_sizes = config.get("batch_size", [64])  # 默认 batch_size = 64
    quant_methods = ["tensorrt_int8", "tensorrt_fp16", "sparsity_fp16", "tensorrt_fp32"]

    results_collector = []

    for bs in batch_sizes:
        for quant_method in quant_methods:
            logger.info(f"Running {model_name} | batch_size={bs} | quant={quant_method} ...")

            # 更新 `config`，修改 `batch_size` 和 `quantization_method`
            config["batch_size"] = bs
            config["passes"] = {quant_method: config["passes"][quant_method]}

            # 执行 transform_graph（⚠️ 不存储模型）
            graph = transform_graph(
                model=model,
                model_info=model_info,
                model_name=model_name,
                data_module=data_module,
                task=task,
                config=config,
                save_dir=None,  # ⚠️ 这里改为 None，不存储文件
                load_name=load_name,
                load_type=load_type,
                accelerator=accelerator,
            )

            # 收集实验结果
            metrics = _extract_results(graph, bs, quant_method)
            results_collector.append(metrics)

    # 仅保存实验结果，不存模型
    _save_results(results_collector, save_dir)
    _plot_curves(results_collector, save_dir)
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


def _extract_results(graph, batch_size, quant_method):
    """
    提取当前 `graph` 的推理时间、准确率和推理能耗
    """
    analysis_res = PASSES["runtime_analysis_pass"](graph, pass_args={"batch_size": batch_size})

    return {
        "quant_method": quant_method,
        "batch_size": batch_size,
        "accuracy": analysis_res.get("Average Accuracy", 0.0),
        "latency": analysis_res.get("Average Latency", 0.0),
        "energy": analysis_res.get("Inference Energy Consumption", 0.0),
    }


def _prepare_cf_args(config, model, model_info, data_module, task, accelerator):
    """
    如果没有 cf_args，就自动生成；如果有就用 config["cf_args"]。
    如果是 vision 模型，就只保留 pixel_values。
    """
    if "cf_args" in config:
        cf_args = config["cf_args"]
    else:
        cf_args = get_cf_args(model_info=model_info, task=task, model=model)

    if model_info.task_type == "vision":
        dummy_in = get_dummy_input(model_info=model_info, data_module=data_module, task=task, device=accelerator)
        if "pixel_values" in dummy_in:
            cf_args = {"pixel_values": dummy_in["pixel_values"]}
        else:
            cf_args = {"pixel_values": torch.randn(1, 3, 224, 224, device=accelerator)}
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

        case _:
            logger.info(f">> {pass_name} pass (generic) ...")
            my_pass = PASSES[pass_name]
            graph, _ = my_pass(graph, pass_args=pass_cfg)
            return graph, results_collector



def _save_results(results_collector, save_dir, model_name):
    """
    保存实验结果到 CSV 文件，并添加 model_name 字段
    """
    csv_filename = f"{model_name}_experiment_results.csv"
    csv_path = save_dir / csv_filename

    with open(csv_path, "w", newline="") as f:
        fieldnames = ["model_name", "quant_method", "batch_size", "latency", "accuracy", "energy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in results_collector:
            d["model_name"] = model_name  # 在数据中添加 model_name
            writer.writerow(d)

    logger.info(f"Results saved to: {csv_path}")


def _plot_curves(results_collector, save_dir, model_name):
    """
    绘制实验曲线：不同 batch_size 的 (latency, accuracy) 关系，并在文件名中加入 model_name
    """
    plt.figure(figsize=(8, 6))

    colors = {
        "tensorrt_int8": "red",
        "tensorrt_fp16": "blue",
        "sparsity_fp16": "green",
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
    
    # 使用 model_name 作为文件名前缀
    plot_filename = f"{model_name}_experiment_plot.png"
    plot_path = save_dir / plot_filename

    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Plot saved to: {plot_path}")
