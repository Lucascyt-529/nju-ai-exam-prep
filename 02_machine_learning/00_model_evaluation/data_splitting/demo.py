"""展示普通与分层的训练/验证/测试划分。"""
import numpy as np
import starter

def show(name, parts, y=None):
    print(name)
    for label, indices in zip(("train", "validation", "test"), parts):
        suffix = "" if y is None else f", labels={y[indices]}"
        print(f"  {label}: indices={indices}{suffix}")
def main() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    try:
        show("普通留出", starter.train_validation_test_split_indices(8, 0.25, 0.25, 7))
        show("分层留出", starter.stratified_split_indices(y, 0.25, 0.25, 7), y)
        print("相同 seed 再运行会得到相同索引。")
    except NotImplementedError as error:
        print("停止展示：", error, sep="")
    except Exception as error:
        print(f"运行到当前步骤时出错：{type(error).__name__}: {error}")
if __name__ == "__main__":
    main()
