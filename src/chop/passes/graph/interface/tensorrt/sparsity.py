from importlib.util import find_spec
from copy import copy, deepcopy
import logging
import torch
import onnx
import numpy as np

logger = logging.getLogger(__name__)

pytorch_quantization_is_installed = False

if find_spec("pytorch_quantization") is None or find_spec("tensorrt") is None:
    def tensorrt_sparsity_interface_pass(graph, pass_args=None):
        raise ImportError("TensorRT not installed. Cannot run sparsity pass.")
else:
    import tensorrt as trt
    # from pytorch_quantization import quant_modules, calib
    # from pytorch_quantization import nn as quant_nn
    # from pytorch_quantization.nn import TensorQuantizer
    # from pytorch_quantization.tensor_quant import QuantDescriptor
    from cuda import cudart

    from chop.passes.graph.utils import (
        get_mase_op,
        get_mase_type,
        get_node_actual_target,
    )
    from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass
    from ...utils import deepcopy_mase_graph
    from ...transforms.tensorrt.quantize.utils import (
        Int8Calibrator,
        prepare_save_path,
        check_for_value_in_dict,
    )

    from chop.passes.utils import register_mase_pass

    @register_mase_pass(
        "tensorrt_sparsity_interface_pass",
        dependencies=["tensorrt", "pycuda", "cuda"],
    )
    def tensorrt_sparsity_interface_pass(graph, pass_args=None):
        """
        1) 进行 2:4 稀疏化
        2) 导出 ONNX
        3) 构建 TensorRT 引擎 (FP16/INT8 + SPARSE_WEIGHTS)
        4) 返回 {trt_engine_path, onnx_path} 给后续 runtime_analysis_pass
        """
        batch_size = pass_args.get("batch_size", 16)  # 默认 batch_size=16
        precision = pass_args.get("precision", "fp16")  # 选择 FP16 或 INT8
        logger.info(f"Using batch_size={batch_size}, precision={precision} for sparsity pass")

        # 1) 先做 2:4 稀疏化
        _apply_2_4_sparsity_to_pytorch(graph.model, pass_args)

        # 2) 导出 ONNX（传入 batch_size）
        onnx_path = _export_to_onnx(graph.model, pass_args, batch_size)

        # 3) 构建 TensorRT 引擎（传入 batch_size 和 precision）
        trt_path = _build_trt_engine(onnx_path, pass_args, batch_size, precision)

        # 4) 返回路径信息
        meta = {"trt_engine_path": trt_path, "onnx_path": onnx_path}
        return graph, meta

    def _apply_2_4_sparsity_to_pytorch(model, pass_args):
        """
        针对 model 的卷积/全连接权重进行 2:4 结构化稀疏, 
        这里只做最简单示范: 
        以4个连续元素为一组, 置其中2个为0.
        (实际中可根据需求, 也可选2个最小/最大等策略)
        """
        logger.info("Applying 2:4 structured sparsity to PyTorch weights ...")
        
        for name, param in model.named_parameters():
            if param.requires_grad and any(k in name for k in ["conv", "fc"]):
                weight_data = param.detach().cpu().numpy()
                flat = weight_data.flatten()
                for i in range(0, flat.size, 4):
                    end = min(i+4, flat.size)
                    if end - i == 4:
                        # **找出最小的2个值并置零**
                        indices = np.argsort(np.abs(flat[i:end]))[:2]  # 取绝对值最小的2个索引
                        flat[i:end][indices] = 0.
                new_w = flat.reshape(weight_data.shape)
                param.data = torch.from_numpy(new_w).to(param.device)
        logger.info("2:4 structured sparsity done.")



    def _export_to_onnx(model, pass_args, batch_size):
        logger.info(f"Exporting model to ONNX for FP16 + SPARSE (batch_size={batch_size})")
        
        from ...transforms.tensorrt.quantize.utils import prepare_save_path

        onnx_path = prepare_save_path(pass_args, method="onnx", suffix="onnx")

        dataloader = pass_args["data_module"].train_dataloader()
        batch = next(iter(dataloader))
        inputs = batch[0][:batch_size]  # ⚡ 修改这里，使用传入的 batch_size

        if isinstance(model, torch.jit.ScriptModule):
            logger.warning("Model is ScriptModule, converting to standard PyTorch module")
            model = model.to(torch.float32)

        torch.onnx.export(
            model.cuda(),
            inputs.cuda(),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,  # 这里保持静态 batch_size
        )

        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError as e:
            raise Exception(f"ONNX Conversion Failed: {e}")

        logger.info(f"ONNX exported => {onnx_path}")
        return onnx_path



    def _build_trt_engine(onnx_path, pass_args, batch_size, precision="fp16"):
        """构建 TensorRT 引擎，支持 FP16 和 INT8"""
        logger.info(f"Building TensorRT engine with {precision.upper()} + SPARSE_WEIGHTS (batch_size={batch_size})")
        
        from ...transforms.tensorrt.quantize.utils import prepare_save_path
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for e in range(parser.num_errors):
                    logger.error(parser.get_error(e))
                raise RuntimeError("Failed to parse ONNX for sparsity")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)

        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)

            # 修正 `Int8Calibrator` 的参数传递
            num_calibration_batches = pass_args.get("num_calibration_batches", 10)
            dataloader = pass_args["data_module"].train_dataloader()
            cache_file = prepare_save_path(pass_args, method='cache', suffix='cache')

            calibrator = Int8Calibrator(
                nCalibration=num_calibration_batches,  # 传入正确的校准 batch 数量
                input_generator=dataloader,  # 传入数据加载器
                cache_file_path=cache_file   # 传入 cache 文件路径
            )
            config.int8_calibrator = calibrator

            # 设置 INT8 计算层
            for idx in range(network.num_layers):
                layer = network.get_layer(idx)
                if layer.type in [trt.LayerType.CONVOLUTION, trt.LayerType.MATRIX_MULTIPLY]:
                    try:
                        layer.precision = trt.int8
                        layer.set_output_type(0, trt.DataType.INT8)
                    except Exception as e:
                        logger.warning(f"Failed to set layer {idx} ({layer.name}) to INT8: {e}")

        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)

        profile.set_shape(
            input_tensor.name,
            (batch_size,) + input_tensor.shape[1:],  
            (batch_size,) + input_tensor.shape[1:],  
            (batch_size,) + input_tensor.shape[1:],  
        )
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError(f"Failed building TRT engine with SPARSE_WEIGHTS & {precision.upper()}")

        trt_path = prepare_save_path(pass_args, method="trt", suffix="trt")
        with open(trt_path, "wb") as f:
            f.write(engine_bytes)

        logger.info(f"Sparse + {precision.upper()} TensorRT engine built => {trt_path}")
        return trt_path
