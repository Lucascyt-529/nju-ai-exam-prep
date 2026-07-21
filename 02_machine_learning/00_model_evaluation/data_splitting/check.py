"""数据划分日常核对。"""
import numpy as np
import starter

def main() -> int:
    passed = 0
    try:
        first = starter.train_validation_test_split_indices(10, 0.2, 0.2, 3)
        second = starter.train_validation_test_split_indices(10, 0.2, 0.2, 3)
        all_indices = np.concatenate(first)
        checks = [("三份互不重叠且覆盖全部样本", len(np.unique(all_indices)) == 10, True), ("固定 seed 可复现", all(np.array_equal(a, b) for a, b in zip(first, second)), True)]
        y = np.array([0] * 10 + [1] * 10); parts = starter.stratified_split_indices(y, 0.2, 0.2, 5)
        checks.append(("分层后每份都含两类", all(set(y[index]) == {0, 1} for index in parts), True))
        checks.append(("训练索引不含验证/测试索引", set(parts[0]).isdisjoint(parts[1]) and set(parts[0]).isdisjoint(parts[2]), True))
        for name, actual, expected in checks:
            ok = actual == expected; print(f"{name}: 期望={expected}, 实际={actual}, 通过={ok}"); passed += ok
    except NotImplementedError as error:
        print("停止核对：", error, sep="")
    except Exception as error:
        print(f"运行错误：{type(error).__name__}: {error}")
    print(f"通过: {passed}/4"); return 0 if passed == 4 else 1
if __name__ == "__main__":
    raise SystemExit(main())
