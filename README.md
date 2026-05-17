# Combating Stochastic Hybrid Attacks for Trustworthy Federated Learning in Mobile Edge Computing

## 项目介绍

本项目为论文官方开源实现，面向移动边缘计算场景下联邦学习的安全性问题，提供自编码器异常检测、连续状态POMDP自适应决策、进化状态子集优化、攻击行为聚类分类与差异化防御全流程功能。框架能够同时防御投毒攻击和搭便车攻击两类随机混合攻击，实现攻击类型识别与差异化处理。

## 安装依赖

```bash
pip install torch torchvision numpy scikit-learn
```

## 使用方法

### 一键运行全流程

```bash
python main.py
```

### 批量实验

```bash
python run_experiment.py
```

## 项目结构

```
pomdpFL/
├── main.py                         # 主入口
├── _main.py                        # 核心训练流程
├── parser.py                       # 参数配置
├── run_experiment.py               # 批量实验脚本
├── clients.py                      # 正常客户端定义
├── clients_attackers.py            # 攻击客户端定义（投毒 / 搭便车）
├── server.py                       # 服务器端聚合与防御逻辑
├── dataloader.py                   # 数据加载（IID / Non-IID 划分）
├── models/                    
│   ├── resnet18.py                 # ResNet18 模型
│   └── mobilenetV2.py              # MobileNetV2 模型
└── utils/
    ├── autoencoder.py              # AutoEncoder 异常检测模块
    ├── pomdp.py                    # 连续状态 POMDP 求解器
    ├── pomdpfl_aggregator.py       # POMDP-FL 聚合器（两阶段框架整合）
    ├── supervised_detector.py      # 细粒度攻击分类器
    └── utils.py                    # 工具函数
```

## 模块说明

### 1. AutoEncoder 异常检测

- **算法**：基于 EMA 梯度的 AutoEncoder 重建误差检测
- **作用**：提取客户端末层梯度的指数移动平均（EMA），通过 AutoEncoder 重建误差识别异常客户端，实现粗粒度异常检测
- **对应文件**：`utils/autoencoder.py`

### 2. 连续状态 POMDP 自适应决策

- **算法**：基于 Fourier 级数信念状态近似 + 自适应粒子滤波 + 进化状态子集优化
- **作用**：将防御激活建模为连续状态 POMDP，根据 FL 系统脆弱性动态决定是否触发防御，平衡防御收益与时间开销
- **对应文件**：`utils/pomdp.py`

### 3. 攻击行为聚类与分类

- **聚类**：聚类将异常客户端划分为两组
- **分类**：多维特征分析区分投毒与搭便车攻击
- **对应文件**：`utils/supervised_detector.py`

### 4. 差异化防御策略

- **投毒客户端**：永久移除，禁止参与后续 FL 训练轮次
- **搭便车客户端**：要求重新提交有效的本地模型更新
- **对应文件**：`server.py`、`utils/pomdpfl_aggregator.py`

### 5. 攻击模型

- **投毒攻击**：标签翻转攻击，将源类标签篡改为目标类标签
- **搭便车攻击**：噪声注入攻击，向全局模型添加高斯噪声伪装为本地更新
- **随机混合攻击**：两类攻击在 FL 训练过程中随机并发，异常客户端以一定概率发起攻击
- **对应文件**：`clients_attackers.py`

## 项目声明

- **项目名称**：Combating Stochastic Hybrid Attacks for Trustworthy Federated Learning in Mobile Edge Computing
- **作者**：Yin Qixi, Cao Kun
- **单位**：暨南大学网络空间安全学院
- **开发语言**：Python
- **代码规模**：约 3400 行
