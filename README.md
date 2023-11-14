## Installation
If you use a CPU environment, please run:
```
pip install -r requirements_cpu.txt
```

## Inference
After the above steps are finished, please check `inference_cpu.py` for an example of making a 12-min weather forecast on CPU with the 12-min model.

For example, running the following command, one can get the 12-min forecast in the `output_data` folder:
```
python inference_cpu.py
```

Also, `inference_iterative.py` shows an example to generate per-12-min forecast within 6 hour.

`inference_iterative1.py` and `inference_iterative2.py` are 

## data preparation
Data should be converted to two numpy .npy file. The first .npy is for node which has a shape **(1, 256002,221)** and the second .npy is for edge with a shape of **(1, 768000, 55)**. The channel order of node is **theta** **rho** **qv** **w**, and the height level order is **0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54(, 55 for w)**. Edge vairiable is **u** with level order **0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55**.

## model download
The model has 5 parts (~5.5GB totally) from Google drive.

The 12-min model (ft12minxxx): https://drive.google.com/drive/folders/1hUS81bGREOOeonVRgjFsjkdDJdfhy_6_

The 1-hour model (ft1hxxx): https://drive.google.com/drive/folders/1hUS81bGREOOeonVRgjFsjkdDJdfhy_6_

put the 12-min model into ```onnx_models/12min``` and 1h model into  ```onnx_models/1h``` and change **model parts** in ```inference_cpu.py``` and ```inference_iterative.py```.