# ImageClarityAssessment：图像清晰度分析

目标：对输入的单张照片进行模糊检测，帮助筛选可能影响后续视觉识别的图片。

## 数据集介绍

使用 CERTH Image Blur Dataset，包含清晰、自然模糊和人工模糊图片。

- 训练集：清晰 630 张、自然模糊 220 张、人工模糊 150 张，用于分析分数和选择阈值。
- 自然模糊评估集：清晰 589 张、模糊 411 张。
- 人工模糊评估集：清晰 30 张、模糊 450 张。

评估集的标签保存在对应 Excel 中，`-1` 表示清晰，`1` 表示模糊。

数据来源：[CERTH 官方页面](https://mklab.iti.gr/results/certh-image-blur-dataset/)。

## 项目实现内容

目前使用 OpenCV 将图片转为灰度图，分别计算拉普拉斯方差和 Tenengrad（Sobel 梯度能量）作为清晰度分数，批量保存 CSV，并绘制分数分布图。

在训练集比较候选阈值后，选择阈值 **40**：分数小于 40 判为模糊，否则判为清晰。固定阈值后，分别测试自然模糊集和人工模糊集，并导出完整预测结果，支持查看错例。

### 拉普拉斯方差阈值选择

训练集的分数统计如下（保留两位小数）：

| 类别 | 数量 | 最小值 | 中位数 | 最大值 |
| --- | ---: | ---: | ---: | ---: |
| 人工模糊 | 150 | 0.71 | 18.95 | 341.97 |
| 自然模糊 | 220 | 3.25 | 45.07 | 512.66 |
| 清晰 | 630 | 3.83 | 128.95 | 2532.14 |

清晰图片的中位数更高，但三类分数范围重叠，不能直接根据最小值、最大值或中位数确定阈值。因此，在训练集比较了 20、40、60、80、100、120、150 七个候选阈值。

其中，阈值 **40** 的训练集准确率最高，为 **75.00%**：模糊被放行 166/370，清晰被误判为模糊 84/630；阈值 20 和 60 的准确率分别为 73.40% 和 73.60%。本轮以候选阈值中的最高训练集准确率为选择标准，将 40 固定为基线阈值，再进行独立评估。它并非所有可能阈值中的最优值，也不是通用阈值。

### Tenengrad 阈值选择

Tenengrad 使用横向与纵向 Sobel 梯度平方和的平均值作为分数。训练集统计如下（保留两位小数）：

| 类别 | 数量 | 最小值 | 中位数 | 最大值 |
| --- | ---: | ---: | ---: | ---: |
| 人工模糊 | 150 | 17.49 | 1259.88 | 13837.91 |
| 自然模糊 | 220 | 56.95 | 853.81 | 15770.83 |
| 清晰 | 630 | 49.62 | 3429.38 | 30820.84 |

分数尺度与拉普拉斯方差不同，因此重新比较了 500、1000、1500、2000、2500、3000、4000 七个候选阈值。其中，阈值 **1000** 的训练集准确率最高，为 **73.50%**：模糊被放行 178/370，清晰被误判为模糊 87/630；阈值 500 和 1500 的准确率分别为 71.90% 和 71.70%。

因此固定阈值 1000：分数小于 1000 判为模糊，否则判为清晰，再在相同的两个评估集上进行方法对照。该阈值仅为本轮候选值中的选择，并非通用阈值。

## 项目目录

```text
ImageClarityAssessment/
├── .gitignore
├── README.md
├── CERTH_ImageBlurDataset/          # 本地数据集及标签
├── src/
│   ├── ObservationOfLaplacianVariance/
│   │   ├── LaplacianVariance.py     # 训练集评分
│   │   ├── analyze_scores.py        # 统计与分布图
│   │   └── evaluate.py              # 评估与错例分析
│   └── Tenengrad/
│       ├── Tenengrad.py             # 训练集评分
│       ├── analyze_scores.py        # 统计与分布图
│       └── evaluate.py              # 评估与错例分析
└── outputs/
    ├── laplacian_variance/          # 拉普拉斯方差结果
    └── tenengrad/                   # Tenengrad 结果
```

### 两种方法测试结果

| 测试集 | 方法（阈值） | 模糊被放行 | 清晰被误判为模糊 | 准确率 |
| --- | --- | ---: | ---: | ---: |
| 自然模糊（1,000 张） | 拉普拉斯方差（40） | 118/411（28.7%） | 116/589（19.7%） | 76.60% |
| 自然模糊（1,000 张） | Tenengrad（1000） | 98/411（23.8%） | 88/589（14.9%） | 81.40% |
| 人工模糊（480 张） | 拉普拉斯方差（40） | 9/450（2.0%） | 8/30（26.7%） | 96.46% |
| 人工模糊（480 张） | Tenengrad（1000） | 19/450（4.2%） | 5/30（16.7%） | 95.00% |

在本轮固定阈值对比中，Tenengrad 在自然模糊集上的准确率提高了 4.8 个百分点，两类误判都减少。人工模糊集上，拉普拉斯方差放行的模糊图更少，而 Tenengrad 误判的清晰图更少，没有一种方法在所有指标上都更好。人工模糊集仅有 30 张清晰图片，仍需结合两类误判理解整体准确率。

## 运行方式

建议使用 Python 3.10 或以上版本。在项目根目录安装依赖：

```powershell
pip install -r requirements.txt
```

从上面的官方页面下载数据集，将其解压到项目根目录，确保存在 `CERTH_ImageBlurDataset/TrainingSet` 和 `CERTH_ImageBlurDataset/EvaluationSet`。图片和原始标签不上传到仓库。

依次运行训练集评分、分数分析和两组测试：

```powershell
python src/ObservationOfLaplacianVariance/LaplacianVariance.py
python src/ObservationOfLaplacianVariance/analyze_scores.py
python src/ObservationOfLaplacianVariance/evaluate.py --dataset NaturalBlurSet
python src/ObservationOfLaplacianVariance/evaluate.py --dataset DigitalBlurSet
```

结果保存在 `outputs/laplacian_variance/`。测试默认不弹出图片窗口；需要逐张查看前十个错例时，在测试命令后加 `--show-errors`。

Tenengrad 使用相同流程，结果保存在 `outputs/tenengrad/`：

```powershell
python src/Tenengrad/Tenengrad.py
python src/Tenengrad/analyze_scores.py
python src/Tenengrad/evaluate.py --dataset NaturalBlurSet
python src/Tenengrad/evaluate.py --dataset DigitalBlurSet
```
