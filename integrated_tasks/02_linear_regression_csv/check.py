"""综合题分阶段核对；未完成时保留已通过结果并友好停止。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import starter

BASE=Path(__file__).resolve().parent; TOTAL=10
def report(name,actual,expected):
    try: ok=bool(np.allclose(actual,expected))
    except (TypeError,ValueError): ok=actual==expected
    print(f"{name}: 期望={expected}, 实际={actual}, 通过={ok}"); return ok

def main():
    passed=0
    try:
        train_ids,X_train,y_train=starter.load_regression_csv(BASE/"data"/"train.csv",has_target=True)
        test_ids,X_test,y_test=starter.load_regression_csv(BASE/"data"/"test.csv",has_target=False)
        passed+=report("1. load_regression_csv(train)",(len(train_ids),X_train.shape,y_train.shape),(6,(6,2),(6,)))
        passed+=report("2. load_regression_csv(test)",(len(test_ids),X_test.shape,y_test),(3,(3,2),None))
        weights,bias=starter.fit_least_squares(X_train,y_train)
        passed+=report("3. fit_least_squares",[*weights,bias],[2.0,3.0,1.0])
        validation_ids,X_validation,y_validation=starter.load_regression_csv(BASE/"data"/"validation.csv",has_target=True)
        prediction=starter.predict(X_validation,weights,bias)
        passed+=report("4. predict",prediction,y_validation)
        metrics=starter.regression_metrics(y_validation,prediction)
        mse=metrics.get("validation_mse",metrics.get("mse")); r2=metrics.get("validation_r2",metrics.get("r2"))
        passed+=report("5. regression_metrics",[mse,r2],[0.0,1.0])
        with TemporaryDirectory() as temp:
            temp=Path(temp); model=temp/"model.npz"; predictions=temp/"predictions.csv"; metrics_path=temp/"metrics.txt"
            starter.save_model(model,weights,bias); loaded_w,loaded_b=starter.load_model(model)
            passed+=report("6. save_model/load_model",[*loaded_w,loaded_b],[2.0,3.0,1.0])
            starter.save_predictions(predictions,test_ids,starter.predict(X_test,loaded_w,loaded_b))
            passed+=report("7. save_predictions",predictions.read_text(encoding="utf-8"),(BASE/"expected"/"predictions.csv").read_text(encoding="utf-8"))
            canonical={"validation_mse":float(mse),"validation_r2":float(r2)}; starter.save_metrics(metrics_path,canonical)
            passed+=report("8. save_metrics",metrics_path.read_text(encoding="utf-8"),(BASE/"expected"/"metrics.txt").read_text(encoding="utf-8"))
            starter.run_pipeline(BASE/"data"/"train.csv",BASE/"data"/"validation.csv",BASE/"data"/"test.csv",predictions,metrics_path,model)
            passed+=report("9. run_pipeline",predictions.exists() and metrics_path.exists() and model.exists(),True)
        passed+=report("10. main 存在且可调用",callable(starter.main),True)
    except NotImplementedError as error: print("停止核对：",error,sep="")
    except Exception as error: print(f"运行错误：{type(error).__name__}: {error}")
    print(f"通过: {passed}/{TOTAL}"); return 0 if passed==TOTAL else 1
if __name__=="__main__": raise SystemExit(main())
