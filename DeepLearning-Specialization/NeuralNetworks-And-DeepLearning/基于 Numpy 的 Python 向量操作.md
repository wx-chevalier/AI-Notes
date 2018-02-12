
# BroadCasting


Numpy 会自动进行矩阵扩展操作以适应指定的矩阵运算


```
A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4


A      (2d array):  5 x 4
B      (1d array):      4
Result (2d array):  5 x 4


A      (3d array):  15 x 3 x 5
B      (3d array):  15 x 1 x 5
Result (3d array):  15 x 3 x 5


A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 5
Result (3d array):  15 x 3 x 5


A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 1
Result (3d array):  15 x 3 x 5


A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
```
```
a = np.random.randn(2, 3) # a.shape = (4, 3)
b = np.random.randn(2, 1) # b.shape = (3, 2)
c = a+b
```
No! In numpy the "*" operator indicates element-wise multiplication. It is different from "np.dot()". If you would try "c = np.dot(a,b)" you would get c.shape = (4, 2).


Also, the broadcasting cannot happen because of the shape of b. b should have been something like (4, 1) or (1, 3) to broadcast properly. So a*b leads to an error!
```
a = np.random.randn(4, 3) # a.shape = (4, 3)
b = np.random.randn(3, 2) # b.shape = (3, 2)
c = a*b


// ValueError: operands could not be broadcast together with shapes (4,3) (3,2)
```


# 延伸阅读