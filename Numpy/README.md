# [Numpy](https://docs.scipy.org/doc/numpy-1.13.0/reference/index.html)

NumPy是使用Python进行科学计算的基础包。 它包含其他内容：

- 强大的多维数组ndarray
- 复杂的（广播）功能
- 用于集成C/C++和Fortran代码的工具
- 实用的线性代数，傅里叶变换和随机数功能

除了用于科学计算，NumPy还可以用作高效的通用数据多维容器。 可以定义任意数据类型。 这使NumPy能够无缝快速地与各种数据库集成。

## [ndarray](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html)

** 定义多维数组 **

``` python
x = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], np.int32)

x.shape
# out: (2, 3)

x.dtype
# out: dtype('int32')

x.ndim
# out: 2

x.size
# out: 6

# 访问第二行第三列
x[1, 2]
# out: 6

# 切片: 访问第二列
x[:, 1]
# out: array([2, 5], dtype=int32)
```

** 方法 **

``` python
x.view(dtype=np.float32)

x.item(5)
# out: 6

x.reshape(3, 2)
# out: array([[1, 2],
#             [3, 4],
#             [5, 6]], dtype=int32)

x.transpose()
# array([[1, 4],
#        [2, 5],
#        [3, 6]], dtype=int32)
# 对一维数值无效，二维数组为矩阵转置，等效于x.T
```
