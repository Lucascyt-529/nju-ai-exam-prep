"""引导演示：隐藏单元逐次加入、冻结和降低残差。"""
import importlib.util
from pathlib import Path
import numpy as np
SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"
def main():
    spec=importlib.util.spec_from_file_location("cascade_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    cc=importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
    X=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.],[0.,-1.],[0.,1.]])
    y=X[:,0]*X[:,1]
    model=cc.fit_cascade_correlation(X,y,n_hidden=3,n_candidates=100,random_state=7)
    print("input width:",X.shape[1])
    print("hidden input widths:",[len(unit["weights"]) for unit in model["hidden_units"]])
    print("output width:",len(model["output_weights"]))
    print("selected correlations:",np.round(model["selected_correlations"],6).tolist())
    print("mse history:",np.round(model["mse_history"],8).tolist())
    print("prediction shape:",cc.predict_cascade(model,X).shape)
if __name__=="__main__": main()
