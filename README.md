# NSCLC

## 概览
- 自监督预训练：使用 OpenMMLab 的 `mmselfsup` 训练 SimCLR、SimSiam、MoCo v3，统一采用 ResNet-18 作为 backbone。
- 预训练产物：`mmselfsup` 导出的 backbone 权重放在 `model/` 目录（按各模型命名），下游分类/分割直接加载。
- 代码结构：数据处理在 `data_pipeline/`，推理与可视化在 `predict_pipeline/`，模型与损失等在 `model/` 与 `utils/`。

## 脱敏说明（需要自行补全的内容）
- 配置文件：`config/` 下所有 JSON（sl2model、ssl2model、map 等）已使用 `/path/to/...` 占位符；使用前请填入真实的 train/valid/test 列表与权重保存路径。
- 预训练权重：`model/msdnet.py` 默认从 `torch.hub.get_dir()/checkpoints/msdnet-step=4-block=5.pth.tar` 读取，可通过环境变量 `MSDNET_PRETRAINED` 指定实际路径。
- 其他路径：`data_pipeline/` 与 `predict_pipeline/` 中示例路径均为占位符，同样需要替换成实际数据与模型位置。
- IDE/缓存：`.idea/`、`__pycache__` 等生成物已清理/覆盖，不含真实路径。

## 使用提示
- 预训练：按照 `mmselfsup` 官方流程复现 SimCLR/SimSiam/MoCo v3（ResNet-18），训练完将 backbone 权重放入 `model/` 或在配置中写入真实路径。
- 下游训练/推理：选择相应的 `config/.../exp*.json`，补全数据与 checkpoint 路径后运行对应脚本。
- 代码中多折交叉验证未更新（[TODO]），数据划分应按照病人划分，即没折中训练和验证的图像块来自不同病人。
