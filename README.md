# vit
华中科技大学人工智能与自动化学院《视觉认知工程》结课作业工作目录。

## 环境配置
建议pytorch 2.1.2以上版本，以及最新版的Transformer库。使用以下命令配置环境。
```sh
pip install -r requirement.txt
```

## 复现
- 按上节配置环境。
- 准备预训练权重与数据集。代码支持自行下载，但考虑到网络条件，可以提前将预训练权重下载至相关文件夹，示例：在hugging face官网下载后放置到`google/vit-base-patch16-224/`文件夹下。
- 配置`config.py`文件，调整超参数。
- 运行`main.py`文件训练，自动评估模型指标。

## 评估
在`eval.py`中已定义评估函数，传入`model`和`testloader`即可运行评估。

## 可视化
运行`visualize.py`可视化相关的注意力矩阵。具体请参照代码注释。

