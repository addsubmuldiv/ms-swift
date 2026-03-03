在NPU运行时，需要安装以下依赖：
- flash-linear-attention
- triton-ascend


## 安装triton-ascend

triton-ascend 可以通过以下命令安装：
```bash
pip install -i https://test.pypi.org/simple/ triton-ascend==3.2.0rc4  # 和CANN 8.3.rc1对应，8.5.0使用3.2.0
```

具体其他历史版本可以参考：https://test.pypi.org/project/triton-ascend/#history

注意：triton-ascend版本和CANN版本需要对应
参考triton-ascend的文档: https://triton-ascend.readthedocs.io/zh-cn/latest/quick_start.html

# 安装flash-linear-attention

flash-linear-attention 可以通过以下命令安装：
```bash
pip install flash-linear-attention
```


