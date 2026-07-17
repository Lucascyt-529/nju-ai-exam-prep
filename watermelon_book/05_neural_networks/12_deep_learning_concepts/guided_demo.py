"""引导演示：逐层编码形状与卷积权共享参数量。"""
import importlib.util
from pathlib import Path
import numpy as np
SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"
def main():
    spec=importlib.util.spec_from_file_location("deep_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    deep=importlib.util.module_from_spec(spec); spec.loader.exec_module(deep)
    X=np.array([[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,1.,0.,0.],[0.,0.,1.,1.]])
    pretrained=deep.greedy_layerwise_pretrain(X,(3,2)); kernels=np.array([[1.,-1.]])
    responses=deep.shared_convolution_1d(X,kernels); counts=deep.convolution_parameter_counts(4,2,1)
    print("code shapes:",[shape for shape in pretrained["code_shapes"]])
    print("layer reconstruction mse:",np.round(pretrained["reconstruction_errors"],8).tolist())
    print("convolution output shape:",responses.shape)
    print("shared/local parameters:",counts["shared"],counts["locally_connected"])
if __name__=="__main__": main()
