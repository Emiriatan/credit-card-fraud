# 信用卡欺诈检测分析计划

## 项目理解

基于Kaggle作者Gabriel Preda的分析方法，这是一个信用卡欺诈检测项目，使用包含284,807笔交易的经典数据集。数据高度不平衡（欺诈交易仅占0.17%，492笔欺诈交易 vs 284,315笔正常交易），特征经过PCA转换保护隐私。

## 核心分析策略

### 数据不平衡处理
- **主要方法**: 使用类别权重和交叉验证策略
- **不采用**: SMOTE等过采样方法（避免信息泄露）
- **关键策略**: 使用分层K折交叉验证确保每个fold中都有足够的欺诈样本

### 模型选择重点
- **主要模型**: RandomForest, AdaBoost, CatBoost, XGBoost, LightGBM
- **基准模型**: 逻辑回归
- **评估标准**: ROC-AUC为主要指标，精确率和召回率平衡考虑

## 详细执行计划

### 阶段1：数据概览与初步检查

**1.1 数据加载与基本信息检查**
- 导入pandas、numpy、matplotlib、seaborn等必要库
- 加载creditcard.csv数据集
- 检查数据维度、列名、数据类型
- 查看前几行和后几行数据

**1.2 数据质量评估**
- 检查缺失值情况
- 检查重复数据
- 验证数据格式一致性
- 统计各列的基本统计信息（均值、标准差、最小值、最大值等）

**1.3 数据不平衡性分析**
- 统计欺诈和非欺诈交易数量
- 计算类别比例
- 可视化类别分布

### 阶段2：探索性数据分析（EDA）

**2.1 数据分布可视化**
- 绘制类别分布饼图和条形图
- 保存为HTML格式：class_distribution.html
- 分析欺诈vs正常交易的比例（492 vs 284,315）

**2.2 交易金额分析**
- 分析Amount列的统计特征
- 比较欺诈和非欺诈交易的金额分布
- 绘制金额分布的箱线图和直方图
- 保存为HTML格式：amount_distribution.html

**2.3 特征相关性分析**
- 计算所有特征之间的相关系数矩阵
- 绘制热力图展示相关性
- 保存为HTML格式：correlation_heatmap.html
- 分析与目标变量（Class）相关性较高的特征

**2.4 关键特征分布**
- 选择Top 10重要特征进行分布分析
- 比较欺诈和非欺诈样本的特征分布差异
- 绘制关键特征的密度图
- 保存为HTML格式：feature_distribution.html

### 阶段3：数据预处理

**3.1 特征工程**
- 对Amount特征进行RobustScaler标准化处理（基于作者方法）
- 不创建时间相关特征（保持原始特征结构）
- 避免复杂的特征工程以保持模型的可解释性

**3.2 数据分割**
- 使用标准train_test_split（不按时间顺序，作者方法）
- 划分训练集（80%）、测试集（20%）
- 使用stratify参数确保类别平衡

**3.3 处理类别不平衡**
- **主要策略**: 使用class_weight='balanced'参数
- **不采用**: SMOTE、ADASYN等过采样方法
- **验证方法**: 使用StratifiedKFold交叉验证

### 阶段4：模型构建与训练

**4.1 核心模型建立（基于作者方法）**
- **逻辑回归**: 基准模型，使用class_weight='balanced'
- **随机森林**: 使用class_weight='balanced'，n_estimators=100
- **AdaBoost**: 重点模型，使用默认参数
- **CatBoost**: 重点模型，使用默认参数
- **XGBoost**: 使用scale_pos_weight处理不平衡
- **LightGBM**: 使用is_unbalance=True参数

**4.2 交叉验证策略**
- 使用StratifiedKFold（n_splits=5）
- 计算每个fold的ROC-AUC分数
- 记录平均性能和标准差
- 绘制交叉验证结果

**4.3 模型训练重点**
- 所有模型都使用相同的交叉验证策略
- 重点训练AdaBoost和CatBoost（作者最佳模型）
- 记录训练时间和预测性能

### 阶段5：模型评估与比较

**5.1 评估指标选择（基于作者方法）**
- **主要指标**: ROC-AUC（作者重点关注的指标）
- **辅助指标**: 精确率、召回率、F1分数
- **可视化工具**: ROC曲线、混淆矩阵
- **不强调**: Precision-Recall AUC（作者未重点使用）

**5.2 模型性能对比**
- 在测试集上评估所有模型
- 绘制所有模型的ROC曲线对比图
- 创建模型性能排行榜（基于ROC-AUC）
- 重点比较AdaBoost和CatBoost的性能差异
- 保存图表为PNG和PDF格式

**5.3 特征重要性分析**
- 分析随机森林的特征重要性
- 对比不同模型的特征重要性排名
- 识别最重要的V1-V28特征
- 可视化Top 10重要特征
- 保存图表为PNG和PDF格式

**5.4 混淆矩阵分析**
- 为每个模型生成混淆矩阵
- 分析误报和漏报情况
- 保存图表为PNG和PDF格式

### 阶段6：结果分析与报告

**6.1 最终HTML总结报告**
- 整合所有分析结果到一个综合HTML报告
- 包含：项目概述、数据探索、模型性能、特征重要性、业务价值
- 添加导航目录和交互式图表
- 保存为：credit_card_fraud_detection_report.html
- 报告内容包括：
  - 数据概览和探索性分析结果
  - 6个机器学习模型的性能对比
  - 最优模型（AdaBoost/CatBoost）的详细分析
  - 特征重要性分析
  - 业务价值评估和建议
  - 改进方向和未来工作建议

## 技术栈（基于作者方法）

**核心库：**
- pandas: 数据处理和分析
- numpy: 数值计算
- matplotlib/seaborn: 数据可视化
- scikit-learn: 机器学习算法和交叉验证
- xgboost: 梯度提升算法
- lightgbm: 轻量级梯度提升
- catboost: 类别提升算法
- sklearn.ensemble: RandomForest和AdaBoost
- plotly: 交互式图表生成
- jinja2: HTML报告模板生成

**特别注意：**
- 不使用imbalanced-learn库（作者未使用SMOTE）
- 不使用shap库（作者未进行深度模型解释）
- 重点使用scikit-learn的内置功能

## 预期成果（基于作者方法）

1. **核心成果**: 6个机器学习模型的ROC-AUC性能对比
2. **关键模型**: AdaBoost和CatBoost的详细分析
3. **特征重要性**: 基于随机森林的特征重要性排名
4. **可视化**: ROC曲线对比图和混淆矩阵
5. **交叉验证**: 5折交叉验证的稳定性评估
6. **最终HTML报告**: 完整的交互式HTML分析报告
7. **图表文件**: 所有可视化图表的PNG和PDF版本
8. **数据文件**: 处理后的数据和模型预测结果

## 作者方法的关键优势

1. **简洁高效**: 避免复杂的特征工程和过采样
2. **实用性强**: 使用标准的机器学习库和方法
3. **结果稳定**: 通过交叉验证确保模型稳定性
4. **可重现性**: 方法简单明确，易于重现
5. **性能优秀**: AdaBoost和CatBoost达到约0.98的ROC-AUC

## 风险与挑战（基于作者经验）

1. **数据不平衡**: 492个欺诈样本vs 284,315个正常样本
2. **特征解释性**: PCA特征V1-V28难以解释具体含义
3. **过拟合风险**: 复杂模型可能在训练集上过拟合
4. **业务平衡**: 需要平衡误报（客户体验）和漏报（金融损失）

## 时间估算（基于作者方法）

- 阶段1-2：数据探索与EDA（1天）
- 阶段3：数据预处理（0.5天）
- 阶段4：模型训练与交叉验证（1-2天）
- 阶段5：模型评估与可视化（0.5天）
- 阶段6：结果分析与报告（1天）

总计：约4-5.5天完成完整分析

## 文件输出结构

```
credit_card_fraud_analysis/
├── data/
│   ├── creditcard.csv (原始数据)
│   ├── processed_data.csv (处理后数据)
│   └── model_predictions.csv (模型预测结果)
├── models/
│   ├── random_forest_model.pkl
│   ├── adaboost_model.pkl
│   ├── catboost_model.pkl
│   ├── xgboost_model.pkl
│   └── lightgbm_model.pkl
├── visualizations/
│   ├── roc_curves_comparison.png
│   ├── feature_importance.png
│   ├── confusion_matrices.png
│   ├── data_distribution.png
│   └── model_performance_table.png
├── credit_card_fraud_detection_report.html (最终HTML报告)
└── analysis_plan.md
```