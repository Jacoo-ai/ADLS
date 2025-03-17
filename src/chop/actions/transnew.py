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


def transnew(
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
    主入口：根据 config 中的 transform.style 不同，调用 transform_graph 或 transform_module。
    """
    accelerator = parse_accelerator(accelerator)
    model = pre_transform_load(load_name, load_type, model)
    model.to(accelerator)

    config = load_config(config)
    style = config["transform"].get("style", "graph")

    if style == "graph":
        logger.info(f"(transnew) Using graph style transform on {model_name} ...")
        transform_graph(
            model=model,
            model_info=model_info,
            model_name=model_name,
            data_module=data_module,
            task=task,
            config=config,
            save_dir=save_dir,
            load_name=load_name,
            load_type=load_type,
            accelerator=accelerator,
        )
    elif style == "module":
        logger.info(f"(transnew) Using module style transform on {model_name} ...")
        transform_module(
            model=model,
            model_info=model_info,
            model_name=model_name,
            data_module=data_module,
            task=task,
            config=config,
            save_dir=save_dir,
            load_name=load_name,
            load_type=load_type,
            accelerator=accelerator,
        )
    else:
        raise ValueError(f"Unsupported transform style {style!r}")


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
    在 FX graph 级别进行 transform，并支持 batch_size = [10,64,128] 的写法。
    关键：在一次 transform 中循环多次，把 data_module.batch_size 改成单个 int 再执行 pass。
    """
    accelerator = parse_accelerator(accelerator)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 读取 batch_size 配置
    # 如果 toml里是 batch_size = 64，就变成 [64]；如果是 [10,64,128]，就直接用
    batch_sizes = config.get("batch_size", 64)
    if not isinstance(batch_sizes, list):
        batch_sizes = [batch_sizes]

    # 用来收集 (accuracy, latency, stage, batch_size) 等结果
    results_collector = []

    # 对每个 batch_size 依次执行 pass
    for bs in batch_sizes:
        logger.info(f"=== Start passes for batch_size={bs} ===")

        # 在这里将 data_module.batch_size 改成当前的单个 int
        data_module.batch_size = bs

        # 每次都重载一下初始模型，以免 pass 改动了它
        current_model = pre_transform_load(load_name, load_type, model)
        current_model.to(accelerator)

        # 做一下 prepare_data() & setup()，让 data_module 的 DataLoader 改成新的 batch_size
        data_module.prepare_data()
        data_module.setup()

        # 根据 config 是否提供 cf_args，生成 forward args
        cf_args = _prepare_cf_args(config, current_model, model_info, data_module, task, accelerator)

        # 构建 MaseGraph
        graph = MaseGraph(model=current_model, cf_args=cf_args)
        graph, _ = init_metadata_analysis_pass(graph, pass_args=None)

        # 如果是 "mz" 类型，就加载图结构
        if load_name is not None and load_type == "mz":
            graph, _ = load_mase_graph_interface_pass(graph, pass_args={"load_dir": load_name})
        else:
            # 否则做 metadata 分析
            dummy_in = get_dummy_input(model_info=model_info, data_module=data_module, task=task, device=accelerator)
            if len(graph.model.additional_inputs) > 0:
                dummy_in = dummy_in | graph.model.additional_inputs
            graph, _ = add_common_metadata_analysis_pass(graph, pass_args={"dummy_in": dummy_in})
            graph, _ = add_software_metadata_analysis_pass(graph, pass_args=None)

        # 给 passes 中的 pass_config 赋值 batch_size=bs
        passes_config = config["passes"]
        for pass_name, pass_cfg in passes_config.items():
            pass_cfg["batch_size"] = bs

        # 依次执行 passes
        graph = _run_all_passes(graph, passes_config, data_module, task, model_info, accelerator, config, results_collector)

        logger.info(f"=== Finished passes for batch_size={bs} ===\n")

    # 全部跑完后，画图
    if results_collector:
        _plot_curves(results_collector, save_dir)

    # 存储最终 graph
    final_ckpt = save_dir / "transformed_ckpt"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})
    save_mase_graph_interface_pass(graph, pass_args=final_ckpt)

    logger.info("transform_graph complete.")
    return graph


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
    from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
    from chop.passes.graph.utils import deepcopy_mase_graph

    match pass_name:
        case "runtime_analysis_ori":
            logger.info(">> runtime_analysis_ori pass ...")
            graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})
            ori_graph = deepcopy_mase_graph(graph)

            _, analysis_res = PASSES["runtime_analysis_pass"](ori_graph, pass_args=pass_cfg)
            if analysis_res is not None:
                acc = analysis_res.get("Average Accuracy", 0.0)
                lat = analysis_res.get("Average Latency", 0.0)
                bs = pass_cfg.get("batch_size", 1)
                results_collector.append({
                    "stage": "Original",
                    "accuracy": acc,
                    "latency": lat,
                    "batch_size": bs
                })
            return graph, results_collector

        case "tensorrt_int8" | "tensorrt_fp16" | "tensorrt_fp32":
            logger.info(f">> {pass_name} pass ...")
            graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})

            # INT8 的 calibrate 在engine_interface中单独进行
            graph, runtime_meta = PASSES["tensorrt_engine_interface_pass"](graph, pass_args=pass_cfg)

            _, analysis_res = PASSES["runtime_analysis_pass"](runtime_meta["trt_engine_path"], pass_args=pass_cfg)
            if analysis_res is not None:
                acc = analysis_res.get("Average Accuracy", 0.0)
                lat = analysis_res.get("Average Latency", 0.0)
                stage = "INT8" if pass_name.endswith("int8") else \
                        ("FP16" if pass_name.endswith("fp16") else "FP32")
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


def _plot_results(results_collector, save_dir):
    """
    将 results_collector 里的结果画到一张散点图。区分 batch_size 不同可用不同颜色。
    """
    if not results_collector:
        return

    plt.figure(figsize=(7,5))
    for item in results_collector:
        bs = item["batch_size"]
        acc = item["accuracy"]
        lat = item["latency"]
        stage = item["stage"]

        # 选颜色
        color = "red"
        if bs == 64:
            color = "blue"
        elif bs == 128:
            color = "green"

        plt.scatter(lat, acc, c=color, label=f"{stage}-bs{bs}")
        plt.text(lat, acc, f"{stage}\nbs={bs}", fontsize=8)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Comparison of different precision modes + multiple batch_size")

    # 去重 legend
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), fontsize=8)

    out_path = save_dir / "multi_bs_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to: {out_path}")

def _plot_curves(results_collector, save_dir):
    """
    将 results_collector 里的结果画到一张图。
    生成四条曲线：Original / INT8 / FP16 / FP32
    横坐标 latency (ms)，纵坐标 accuracy。
    在每个点上标注 "bs=??"。
    """

    import matplotlib.pyplot as plt
    import csv

    if not results_collector:
        return

    # 先把结果按 stage 分组
    # stages = ["Original", "INT8", "FP16", "FP32"]
    stage_groups = {}  # key: stage, value: list of dict(item)
    for item in results_collector:
        stage = item["stage"]  # e.g. "Original", "INT8", "FP16", "FP32"
        if stage not in stage_groups:
            stage_groups[stage] = []
        stage_groups[stage].append(item)

    # 为了画曲线，需要对每个 stage 的 (latency, accuracy) 做排序
    # 一般我们想按照 batch_size 升序连接起来
    # 也可以改成按 latency 升序，随你需求
    for stage, items in stage_groups.items():
        # sort by batch_size (这样曲线会按 batch_size 的顺序走)
        items.sort(key=lambda x: x["batch_size"])

    # 给每个 stage 选一个颜色
    color_map = {
        "Original": "red",
        "INT8": "green",
        "FP16": "blue",
        "FP32": "orange",
    }

    plt.figure(figsize=(8,6))

    # 把4种 stage 分别画出来
    for stage, items in stage_groups.items():
        # 准备 x, y
        xs = [d["latency"] for d in items]
        ys = [d["accuracy"] for d in items]
        # 画线
        c = color_map.get(stage, "black")  # 如果没匹配上就用黑色
        plt.plot(xs, ys, marker="o", color=c, label=f"{stage}")

        # 在每个点旁边标注“bs=?”
        for d in items:
            plt.text(d["latency"], d["accuracy"], f"bs={d['batch_size']}", fontsize=4)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Comparison of 4 Precision Modes vs Batch Size")

    plt.legend()
    out_path = save_dir / "multi_bs_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved to: {out_path}")

    # 5) 同时把 results_collector 保存到 CSV
    csv_path = save_dir / "multi_bs_results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["stage", "batch_size", "latency", "accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in results_collector:
            writer.writerow({
                "stage": d["stage"],
                "batch_size": d["batch_size"],
                "latency": d["latency"],
                "accuracy": d["accuracy"],
            })

    logger.info(f"CSV saved to: {csv_path}")