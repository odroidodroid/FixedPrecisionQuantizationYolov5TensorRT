| Name | Data Type | Input Shapes | torch2trt kwargs | Max Error | Throughput (PyTorch) | Throughput (TensorRT) | Latency (PyTorch) | Latency (TensorRT) |
|------|-----------|--------------|------------------|-----------|----------------------|-----------------------|-------------------|--------------------|
|                     torch2trt.tests.torchvision.classification.alexnet | float16 |        [(1, 3, 224, 224)] | {'fp16_mode': True} | 9.16E-05 | nan | 0.00E+00 | 1.2e+03 | 1.68e+03 | 2.09 | 2.57 |
|               torch2trt.tests.torchvision.classification.squeezenet1_0 | float16 |        [(1, 3, 224, 224)] | {'fp16_mode': True} | 4.88E-04 | 80.74 | 1.19E-08 | 408 | 2.37e+03 | 2.95 | 2.51 |
|               torch2trt.tests.torchvision.classification.squeezenet1_1 | float16 |        [(1, 3, 224, 224)] | {'fp16_mode': True} | 4.88E-04 | 79.71 | 1.39E-08 | 300 | 2.86e+03 | 2.77 | 2.48 |
|                    torch2trt.tests.torchvision.classification.resnet18 | float16 |        [(1, 3, 224, 224)] | {'fp16_mode': True} | 9.77E-03 | 67.51 | 5.08E-06 | 356 | 1.91e+03 | 3.16 | 2.56 |
|                    torch2trt.tests.torchvision.classification.resnet34 | float16 |        [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.50E-01 | 66.11 | 3.54E-03 | 244 | 1.14e+03 | 3.95 | 1.98 |
| torch2trt.tests.torchvision.classification.resnet50 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | N/A | N/A | N/A | N/A | N/A |
