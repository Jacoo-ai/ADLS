import tensorrt as trt

TRT_FILE = "/workspace/ADLS_Proj/mase_output/tensorrt/quantization/resnet18_cls_cifar10_2025-03-16/2025-03-16/version_2/model.trt"


with open(TRT_FILE, "rb") as f:
    runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
    engine = runtime.deserialize_cuda_engine(f.read())

# 打印有多少个 IO Tensor
n_io = engine.num_io_tensors
print("Number of IO tensors:", n_io)

for i in range(n_io):
    # 1) 拿到这个index对应的tensor名字
    tensor_name = engine.get_tensor_name(i)
    
    # 2) 再用tensor名字获取其他信息
    tensor_dtype = engine.get_tensor_dtype(tensor_name)
    tensor_shape = engine.get_tensor_shape(tensor_name)
    tensor_format = engine.get_tensor_format(tensor_name)
    tensor_location = engine.get_tensor_location(tensor_name)
    # 也可以看 engine.get_tensor_profile_shape(tensor_name, profileIndex=...) 之类
    
    print(f"[Tensor {i}] name={tensor_name}, dtype={tensor_dtype}, shape={tensor_shape}, format={tensor_format}, location={tensor_location}")

def inspect_trt_engine(trt_file):
    # 1) 反序列化 engine
    with open(trt_file, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        engine = runtime.deserialize_cuda_engine(f.read())

    # 2) 创建 inspector，获取信息 (JSON 或字符串)
    inspector = engine.create_engine_inspector()
    layer_info_str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    
    # 3) 先把原始字符串打印出来
    print("=== Inspector Raw Output ===")
    print(layer_info_str)

    # 4) 尝试解析
    # 如果返回的确实是 JSON 格式，可以:
    #     layer_info = json.loads(layer_info_str)
    #     # 然后看看 layer_info["Layers"] 之类
    #     # 但许多版本只返回字符串数组
    #     # 如果 parse 报错，则只能做字符串方式查找

    # 在许多 TRT 版本里, 可能是一大段带换行的字符串
    # 这里我们就做最简单的: 判断 "INT8" 是否在字串之中
    if "INT8" in layer_info_str.upper():
        print(">>> Found 'INT8' in the inspector output, indicating int8 kernel usage!")
    else:
        print(">>> Did NOT find 'INT8' in the inspector output. Possibly FP32/FP16 or fallback.")

inspect_trt_engine(TRT_FILE)