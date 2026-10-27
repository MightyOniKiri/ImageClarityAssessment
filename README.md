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

目前使用 OpenCV 将图片转为灰度图，计算拉普拉斯方差作为清晰度分数，批量保存 CSV，并绘制分数分布图。

在训练集比较候选阈值后，选择阈值 **40**：分数小于 40 判为模糊，否则判为清晰。固定阈值后，分别测试自然模糊集和人工模糊集，并导出预测结果与错例。

## 项目目录

```text
ImageClarityAssessment/
├── .gitignore
├── README.md
├── CERTH_ImageBlurDataset/          # 本地数据集及标签
├── src/
│   └── ObservationOfLaplacianVariance/
│       ├── LaplacianVariance.py     # 训练集评分
│       ├── analyze_scores.py        # 统计与分布图
│       └── evaluate.py              # 评估与错例分析
└── outputs/
    └── laplacian_variance/          # 分数、测试结果、错例及图表
```

### 基线测试结果

| 测试集 | 图片数 | 模糊被放行 | 清晰被误判为模糊 | 准确率 |
| --- | ---: | ---: | ---: | ---: |
| 自然模糊 | 1,000 | 118/411（28.7%） | 116/589（19.7%） | 76.60% |
| 人工模糊 | 480 | 9/450（2.0%） | 8/30（26.7%） | 96.46% |

当前方法对人工模糊的拦截效果更好，但自然模糊与清晰图片的分数存在较多重叠。人工模糊集的清晰图片较少，不能仅凭整体准确率判断效果，仍需关注两类误判。



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
