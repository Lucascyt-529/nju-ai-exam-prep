# 从空文件重写

完成 starter 后，关闭参考实现并新建临时练习文件。只根据以下契约重写：

```text
predict(X, w, b) -> prediction
mean_squared_error(y_true, y_pred) -> float
mse_gradients(X, y, w, b) -> gradient_w, gradient_b
fit_gradient_descent(...) -> w, b, loss_history
fit_least_squares(X, y, fit_intercept=True) -> w, b
```

本阶段只用 3 个手算样本验证预测、损失、梯度、参数更新和损失下降。输入校验与 shape 专项以后再练。
