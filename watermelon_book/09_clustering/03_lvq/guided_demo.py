"""引导演示：LVQ正确靠近、错误远离与训练历史。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION=Path(__file__).resolve().parent/"reference"/"solution.py"

def main():
    spec=importlib.util.spec_from_file_location("lvq_demo",SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    lvq=importlib.util.module_from_spec(spec); spec.loader.exec_module(lvq)
    prototypes=np.array([[0.,0.],[4.,4.]]); labels=np.array([0,1]); x=np.array([1.,0.])
    toward,winner,correct=lvq.lvq_update(prototypes,labels,x,0,.5); away,_,wrong=lvq.lvq_update(prototypes,labels,x,1,.5)
    print("winner/correct:",winner,correct); print("toward:",toward.tolist()); print("away:",away.tolist(),wrong)
    X=np.array([[0.,0.],[0.,1.],[1.,0.],[4.,4.],[4.,5.],[5.,4.]]); y=np.array([0,0,0,1,1,1])
    model=lvq.fit_lvq(X,y,epochs=10,random_state=3)
    print("history shape:",model["history"].shape); print("prototype labels:",model["prototype_labels"].tolist())
    print("training prediction:",lvq.predict(model,X).tolist())

if __name__=="__main__": main()
