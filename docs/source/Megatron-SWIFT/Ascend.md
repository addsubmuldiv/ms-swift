# Ascend NPU

关于Megatron-SWIFT在Ascend NPU上的环境准备，请参考[NPU最佳实践](../BestPractices/NPU-support.md)。

## NPU 性能数据采集

Megatron-SWIFT 已支持通过命令行参数调用 `torch_npu.profiler.profile` 采集 NPU 性能数据，不需要再手动修改训练循环。采集参数、schedule 语义和示例命令请参考 [NPU最佳实践中的 Megatron-SWIFT NPU Profiling](../BestPractices/NPU-support.md#megatron-swift-npu-profiling)。

## NPU 精度数据采集
### 安装msprobe
```shell
pip install mindstudio-probe
```

### 代码修改
为了支持 msprobe 工具进行精度调试，我们需要修改 `swift/megatron/model/mm_gpt_model.py` 文件中的 `_patch_word_embeddings` 函数。主要改动是调整函数参数和内部实现逻辑，使其能够正确地对嵌入层进行patch

下面是具体的修改内容：

修改前：
```python
def _patch_word_embeddings(self, kwargs):
    origin_forward = VocabParallelEmbedding.forward

    def forward(_self, input_):
        args = get_args()
        reduce_scatter_embeddings = _self.reduce_scatter_embeddings
        _self.reduce_scatter_embeddings = False
        input_ = torch.masked_fill(input_, input_ < 0, 0)
        res = origin_forward(_self, input_)
        _self.reduce_scatter_embeddings = reduce_scatter_embeddings
        packed_seq_params = kwargs.get('packed_seq_params')
        # ...其他逻辑...
        return res
    VocabParallelEmbedding.forward = forward
    try:
        yield
    finally:
        VocabParallelEmbedding.forward = origin_forward

def forward(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    decoder_input: torch.Tensor = None,
    labels: torch.Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    **kwargs,
) -> torch.Tensor:
    if decoder_input is not None:
        pass
    elif self.pre_process:
        kwargs.update({'input_ids': input_ids, 'packed_seq_params': packed_seq_params})
        with self._patch_word_embeddings(kwargs):
            decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)

    # ...其他逻辑...
```

修改后：
```python
def _patch_word_embeddings(self, kwargs, emb):          # 修改1
    origin_forward = emb.word_embeddings.forward        # 修改2

    def forward(input_):                                # 修改3
        args = get_args()
        _self = emb.word_embeddings                     # 修改4
        reduce_scatter_embeddings = _self.reduce_scatter_embeddings
        _self.reduce_scatter_embeddings = False
        input_ = torch.masked_fill(input_, input_ < 0, 0)
        res = origin_forward(input_)                    # 修改5
        _self.reduce_scatter_embeddings = reduce_scatter_embeddings
        packed_seq_params = kwargs.get('packed_seq_params')
        # ...其他逻辑...
        return res

    emb.word_embeddings.forward = forward               # 修改6
    try:
        yield
    finally:
        emb.word_embeddings.forward = origin_forward    # 修改7

def forward(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    decoder_input: torch.Tensor = None,
    labels: torch.Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    **kwargs,
) -> torch.Tensor:
    if decoder_input is not None:
        pass
    elif self.pre_process:
        kwargs.update({'input_ids': input_ids, 'packed_seq_params': packed_seq_params})
        with self._patch_word_embeddings(kwargs, self.language_model.embedding):                # 修改8
            decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)

    # ...其他逻辑...
```

主要变化包括：
1. `_patch_word_embeddings` 方法增加了 `emb` 参数，用于接收 embedding 模块实例
2. 直接获取 `emb.word_embeddings.forward` 而不是 `VocabParallelEmbedding.forward`
3. 内部 `forward` 函数签名从 `(_self, input_)` 改为 `(input_)`
4. 在函数内部通过 `emb.word_embeddings` 获取 `_self`
5. 调用原始 forward 时直接传入 `input_`
6. 使用 `emb.word_embeddings.forward` 进行替换和恢复操作（修改6、7）
7. 在调用 `_patch_word_embeddings` 时传入 `self.language_model.embedding` 实例

对文件swift/megatron/trainers/base.py中的train_step函数进行修改
修改前：
```python
def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, *args,
               **kwargs):
    new_data_iterator = self._replace_data_iterator(data_iterator, model)
    return self._origin_train_step(forward_step_func, new_data_iterator, model, optimizer, opt_param_scheduler,
                                   config, *args, **kwargs)

```
修改后：
```python
def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, *args,
               **kwargs):
    new_data_iterator = self._replace_data_iterator(data_iterator, model)
    from msprobe.pytorch import PrecisionDebugger
    debugger = PrecisionDebugger(dump_path='./dump_path', level='mix', model=model)
    debugger.start()
    try:
        origin_train_step_out = self._origin_train_step(
            forward_step_func, new_data_iterator, model, optimizer, opt_param_scheduler,config, *args, **kwargs)
    finally:
        debugger.stop()
        debugger.step()
    return origin_train_step_out
```


### 使能

另外，由于msprobe不支持融合计算，需要在启动脚本添加`--bias_dropout_fusion false`、`--bias_swiglu_fusion false`、`--cross_entropy_loss_fusion false`

#### 示例
```shell
PYTORCH_NPU_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --mcore_model Qwen2.5-7B-Instruct-mcore \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --tensor_model_parallel_size 2 \
    ...
    --bias_dropout_fusion false \
    --bias_swiglu_fusion false \
    --cross_entropy_loss_fusion false
```
