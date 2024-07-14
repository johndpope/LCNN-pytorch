# LCNN-pytorch
lookup-based convolutional neural network (blast from the past)

https://patents.google.com/patent/US20200364499A1



```shell

Conv2d - Avg time: 0.72 ms, Memory usage: 74.06 MB
LookupConv2d - Avg time: 12.61 ms, Memory usage: 61.56 MB
Speed difference: 0.06x
Memory difference: 1.20x
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                           LookupConv2d         0.00%       0.000us         0.00%       0.000us       0.000us      22.063ms        57.95%      22.063ms      22.063ms             1  
                                           LookupConv2d         1.19%     315.110us        43.64%      11.599ms      11.599ms       0.000us         0.00%      11.493ms      11.493ms             1  
                                   SparseConv2dFunction         0.19%      49.209us         0.25%      67.417us      67.417us      10.597ms        27.83%      10.627ms      10.627ms             1  
void sparse_conv2d_forward_kernel<float>(float const...         0.00%       0.000us         0.00%       0.000us       0.000us      10.597ms        27.83%      10.597ms      10.597ms             1  
                                                 Conv2d         0.00%       0.000us         0.00%       0.000us       0.000us       3.287ms         8.63%       3.287ms       3.287ms             1  
                                            aten::copy_         0.14%      38.303us        14.81%       3.937ms     656.147us       1.782ms         4.68%       1.782ms     296.994us             6  
                                               aten::to         4.10%       1.091ms        18.98%       5.044ms       2.522ms       0.000us         0.00%       1.777ms     888.295us             2  
                                         aten::_to_copy         0.11%      28.041us        14.87%       3.954ms       1.977ms       0.000us         0.00%       1.777ms     888.295us             2  
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us       1.777ms         4.67%       1.777ms     888.295us             2  
                                                 Conv2d         0.73%     194.318us         9.55%       2.538ms       2.538ms       0.000us         0.00%       1.233ms       1.233ms             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 26.581ms
Self CUDA time total: 38.076ms


```

```shell
/take2
python setup.py install
python train.py
```
this sonnect implementation does reduce memory by 20% - but not showing improvements in speed.

```shell
python inference.py
Loaded best model from epoch 2 with validation loss: 0.5942
5_o_Clock_Shadow: False
Arched_Eyebrows: False
Attractive: False
Bags_Under_Eyes: False
Bald: False
Bangs: False
Big_Lips: True
Big_Nose: False
Black_Hair: False
Blond_Hair: False
Blurry: False
Brown_Hair: False
Bushy_Eyebrows: False
Chubby: False
Double_Chin: False
Eyeglasses: False
Goatee: False
Gray_Hair: False
Heavy_Makeup: False
High_Cheekbones: False
Male: False
Mouth_Slightly_Open: False
Mustache: False
Narrow_Eyes: False
No_Beard: True
Oval_Face: False
Pale_Skin: False
Pointy_Nose: False
Receding_Hairline: False
Rosy_Cheeks: False
Sideburns: False
Smiling: False
Straight_Hair: False
Wavy_Hair: False
Wearing_Earrings: False
Wearing_Hat: False
Wearing_Lipstick: False
Wearing_Necklace: False
Wearing_Necktie: False
Young: True
```


**Tensorflow port - vanilla**
```shell
take6/train.py
```