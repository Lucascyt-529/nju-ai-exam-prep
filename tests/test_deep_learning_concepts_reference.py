import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"05_neural_networks"/"12_deep_learning_concepts"
spec=importlib.util.spec_from_file_location("deep_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def data(): return np.array([[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,1.,0.,0.],[0.,0.,1.,1.]])
def test_single_layer_shapes_center_and_reconstruction():
    X=data(); layer=solution.fit_linear_autoencoder_layer(X,2); assert layer["mean"].shape==(4,); assert layer["components"].shape==(2,4); assert layer["code"].shape==(4,2); assert layer["reconstructed"].shape==X.shape; np.testing.assert_allclose(layer["code"].mean(axis=0),0,atol=1e-12)
def test_components_orthonormal_and_ratio_valid():
    layer=solution.fit_linear_autoencoder_layer(data(),3); np.testing.assert_allclose(layer["components"]@layer["components"].T,np.eye(3),atol=1e-12); assert np.all(layer["explained_variance_ratio"]>=0) and layer["explained_variance_ratio"].sum()<=1+1e-12
def test_full_width_reconstructs_exactly():
    layer=solution.fit_linear_autoencoder_layer(data(),4); np.testing.assert_allclose(layer["reconstructed"],data(),atol=1e-12); assert float(layer["reconstruction_mse"])<1e-24
def test_greedy_shapes_and_encode_reproduces_code():
    X=data(); model=solution.greedy_layerwise_pretrain(X,(3,2)); assert model["code_shapes"]==((4,3),(4,2)); assert model["reconstruction_errors"].shape==(2,); np.testing.assert_allclose(solution.encode_layers(X,model["layers"]),model["code"])
def test_each_layer_uses_own_feature_width():
    model=solution.greedy_layerwise_pretrain(data(),(3,2)); assert model["layers"][0]["mean"].shape==(4,); assert model["layers"][1]["mean"].shape==(3,)
def test_pretraining_repeatable_and_input_not_modified():
    X=data(); X0=X.copy(); a=solution.greedy_layerwise_pretrain(X,(3,2)); b=solution.greedy_layerwise_pretrain(X,(3,2)); np.testing.assert_allclose(a["code"],b["code"]); np.testing.assert_array_equal(X,X0)
def test_convolution_hand_values_shape_and_filter_axis():
    X=np.array([[1.,2.,4.,7.]]); kernels=np.array([[1.,-1.],[1.,1.]]); output=solution.shared_convolution_1d(X,kernels); assert output.shape==(1,3,2); np.testing.assert_allclose(output[0,:,0],[-1,-2,-3]); np.testing.assert_allclose(output[0,:,1],[3,6,11])
def test_weight_sharing_translation_response_moves():
    kernel=np.array([[1.,-1.]]); a=solution.shared_convolution_1d(np.array([[1.,0.,0.,0.]]),kernel); b=solution.shared_convolution_1d(np.array([[0.,1.,0.,0.]]),kernel); assert np.argmax(np.abs(b[0,:,0]))>=np.argmax(np.abs(a[0,:,0]))
def test_parameter_counts_hand_values_and_reduction():
    counts=solution.convolution_parameter_counts(10,3,4); assert counts=={"shared":12,"locally_connected":96}; assert counts["shared"]<counts["locally_connected"]
def test_one_position_counts_equal(): assert solution.convolution_parameter_counts(3,3,2)=={"shared":6,"locally_connected":6}
def test_bad_hidden_dimensions_and_layer_parameters_rejected():
    X=data()
    with pytest.raises(ValueError): solution.greedy_layerwise_pretrain(X,())
    with pytest.raises(ValueError): solution.fit_linear_autoencoder_layer(X,5)
    with pytest.raises(ValueError): solution.encode_layers(X,({"mean":np.zeros(3),"components":np.eye(3)},))
def test_bad_convolution_sizes_rejected():
    with pytest.raises(ValueError): solution.shared_convolution_1d(data(),np.ones((1,5)))
    with pytest.raises(ValueError): solution.convolution_parameter_counts(3,4,1)
def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "code shapes: [(4, 3), (4, 2)]" in result.stdout and "convolution output shape: (4, 3, 1)" in result.stdout and "shared/local parameters: 2 6" in result.stdout
