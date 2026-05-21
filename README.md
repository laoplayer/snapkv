## 项目简介
本项目选用NVIDIA开源仓库kvpress中内置的原版snapkv算法，基于Pythia-70M模型完成推理加速实验。本方案属于无训练优化方式，仅在模型推理阶段对KV缓存进行剪枝压缩，无需对模型权重进行微调训练。

## 运行环境
Python 3.10及以上版本

## 部署与运行
1. 安装项目所需依赖
```bash
pip install -r requirements.txt
```
2. 运行主程序执行实验
```bash
python main.py
```
## 实验配置
1. 实验模型：EleutherAI/pythia-70m
2. 优化算法：SnapKV
3. 压缩比例：0.5
4. Pytorch版本：cuda13.0
5. 显卡：RTX5060
6. 在wikitext数据集上进行ppl测试和加速测试
7. 算法来源：https://github.com/NVIDIA/kvpress

## 实验结果说明
程序运行后会自动对比两组推理数据：
1. 原始模型无压缩推理耗时
2. 引入 SnapKV 压缩后的推理耗时
3. 自动计算得出整体加速倍率
实验结论：SnapKV 能够在几乎不影响文本生成质量的前提下，削减冗余KV缓存占用，有效加快大语言模型文本生成速度。

## 项目说明
本项目全程直接调用官方封装好的SnapKVPress接口，未私自修改算法底层源码，严格使用仓库原生提供的算法完成本次课程作业。
