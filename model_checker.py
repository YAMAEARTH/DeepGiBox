import onnx
onnx_model = onnx.load("/home/earth/Documents/pun/DeepGiBox/configs/model/gim_model_decrypted.onnx")
print(onnx.checker.check_model(onnx_model))