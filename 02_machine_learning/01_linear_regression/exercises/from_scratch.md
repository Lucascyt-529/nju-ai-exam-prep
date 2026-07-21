# 从空文件重写

完成 starter 后，关闭参考实现并新建临时练习文件。只根据以下契约重写：

```text
predict(X, w, b) -> prediction
mean_squared_error(y_true, y_pred) -> float
mse_gradients(X, y, w, b) -> gradient_w, gradient_b
fit_gradient_descent(...) -> w, b, loss_history
fit_least_squares(X, y, fit_intercept=True) -> w, b
```

先用单特征精确直线的3个手算样本验证预测、损失、梯度、参数更新和损失下降，
再独立构造一个双特征样例。输入校验以后再练，但必须保持本专题约定的一维标签和预测。

重写完成后，用同一数据分别运行梯度下降与最小二乘，比较参数、最终 MSE 和所需设置。
