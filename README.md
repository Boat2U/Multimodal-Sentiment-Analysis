# Multimodal-Sentiment-Analysis

多模态情感分析————基于Bert和Resnet-152的多种融合方法

## Set up

```
numpy==1.20.3
pandas==1.3.4
Pillow==9.2.0
Pillow==10.0.0
scikit_learn==1.1.1
torch==1.11.0+cu113
transformers==4.18.0
pyecharts==2.0.3
```

你可以通过以下命令一键配置依赖包
```bash
pip install -r requirements.txt
```

## Repository structure
以下是该仓库的结构，仅附上重要文件及解释说明

```
.
├─data
├─images
│  ├─note
│  └─results
├─label
├─models
│  ├─early_fusion.py
│  ├─hybrid_fusion.py
│  └─late_fusion.py
├─output
├─tf_logs                   tensorboard仪表盘记录
├─main.py                   主函数    
├─data_utils.py             数据处理模块
├─README.md
├─results.ipynb             实验结果部分可视化
└─requirements.txt
```

## Fusion type
**Early_fusion**：早期融合
**Late_fusion**：晚期融合
**Hybrid_fusion**：注意力融合（实际也是混合融合）

## Run command

**1. 查看可调整参数**
```bash
python main.py -h
```

**2. 训练**
使用默认参数
```bash
python main.py
```

更改参数训练:
- 多模态融合模型
```bash
python main.py --lr 5e-5 --weight_decay 1e-2 --epochs 10 --seed 233 --fusion_type early_fusion
```
- 消融实验(仅图像)
```bash
python main.py --lr 5e-5 --weight_decay 1e-2 --epochs 10 --seed 233 --fusion_type early_fusion --image_only
```
- 消融实验(仅文本)
```bash
python main.py --lr 5e-5 --weight_decay 1e-2 --epochs 10 --seed 233 --fusion_type early_fusion --text_only
```

**3. 测试**
```bash
python main.py --do_test
```

**4. 启动tensorboard查看图片**
```bash
tensorboard --logdir=tf_logs
```

## Results

| Model | both_acc | image_only_acc | text_only_acc |
| :----: | :----: | :----: | :----: |
| early_fusion | 78.00% | 67.00% | 72.25% |
| late_fusion | 75.25% | 65.50% | 73.50% |
| hybrid_fusion | 74.00% | 65.50% | 73.50% |

## Attribution

- [YeexiaoZheng / Multimodal-Sentiment-Analysis](https://github.com/YeexiaoZheng/Multimodal-Sentiment-Analysis)
- [liyunfan1223 / multimodal-sentiment-analysis](https://github.com/YeexiaoZheng/Multimodal-Sentiment-Analysis)
