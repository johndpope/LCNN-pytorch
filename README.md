# LCNN-pytorch
lookup-based convolutional neural network (blast from the past)

https://patents.google.com/patent/US20200364499A1



```shell

Conv2d - Avg time: 0.38 ms, Memory usage: 74.06 MB
LookupConv2d - Avg time: 10.59 ms, Memory usage: 102.08 MB
Speed difference: 0.04x
Memory difference: 0.73x
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                           LookupConv2d         0.00%       0.000us         0.00%       0.000us       0.000us      12.635ms        45.89%      12.635ms      12.635ms             1  
                                           LookupConv2d         0.57%      89.858us        81.37%      12.793ms      12.793ms       0.000us         0.00%      12.007ms      12.007ms             1  
                                   LookupConv2dFunction         0.96%     151.588us        74.83%      11.765ms      11.765ms      10.951ms        39.78%      11.160ms      11.160ms             1  
                                  lookup_conv2d_forward         0.00%       0.000us         0.00%       0.000us       0.000us      10.951ms        39.78%      10.951ms      10.951ms             1  
                                            aten::copy_         0.23%      36.844us        13.06%       2.053ms     684.330us       1.836ms         6.67%       1.836ms     612.148us             3  
                                               aten::to         6.29%     988.199us        19.41%       3.052ms       1.526ms       0.000us         0.00%       1.803ms     901.614us             2  
                                         aten::_to_copy         0.20%      31.978us        13.13%       2.064ms       1.032ms       0.000us         0.00%       1.803ms     901.614us             2  
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us       1.803ms         6.55%       1.803ms     901.614us             2  
                                                 Conv2d         0.00%       0.000us         0.00%       0.000us       0.000us       1.607ms         5.84%       1.607ms       1.607ms             1  
                                                 Conv2d         1.35%     212.988us        18.61%       2.926ms       2.926ms       0.000us         0.00%       1.283ms       1.283ms             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 15.722ms
Self CUDA time total: 27.532ms

```


this sonnect implementation basically didn't work....