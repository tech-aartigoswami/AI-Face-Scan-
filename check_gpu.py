
import sys
import onnxruntime as ort

print("Python executable:", sys.executable)
print("ONNX Runtime version:", ort.__version__)
print("ONNX Runtime location:", ort.__file__)

try:
    ort.preload_dlls(directory="")
except Exception as e:
    print("DLL preload:", e)

providers = ort.get_available_providers()

print("Available providers:", providers)

if "CUDAExecutionProvider" in providers:
    print("\nSUCCESS: CUDA GPU AVAILABLE!")
else:
    print("\nERROR: CPU VERSION IS ACTIVE!")