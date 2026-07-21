"""逐步展示 CSV 切分和流水线中间量，只调用学生 starter。"""
from pathlib import Path
import starter

BASE=Path(__file__).resolve().parent

def show_head(path):
    lines=path.read_text(encoding="utf-8").splitlines()
    print(path.name,"前两行:")
    for line in lines[:2]: print(" ",line)

def main():
    train=BASE/"data"/"train.csv"; test=BASE/"data"/"test.csv"
    show_head(train); show_head(test)
    try:
        train_ids,X_train,y_train=starter.load_regression_csv(train,has_target=True)
        test_ids,X_test,y_test=starter.load_regression_csv(test,has_target=False)
        print("train 切分: ids=",train_ids[:2],"X shape=",X_train.shape,"y shape=",y_train.shape)
        print("test 切分: ids=",test_ids[:2],"X shape=",X_test.shape,"y=",y_test)
        weights,bias=starter.fit_least_squares(X_train,y_train)
        print("拟合参数: weights=",weights,"bias=",bias)
        print("测试预测前两项:",starter.predict(X_test,weights,bias)[:2])
    except NotImplementedError as error: print("停止展示：",error,sep="")
    except Exception as error: print(f"运行到当前步骤时出错：{type(error).__name__}: {error}")
if __name__=="__main__": main()
