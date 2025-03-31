# 随机森林分类器训练与预测系统


## 项目概述

本项目基于  **JSONL 格式数据** ，通过随机森林算法实现分类任务。支持从训练数据中训练模型、预测测试数据，并将预测结果保存为结构化文件。适用于需要快速搭建分类模型的场景（如文本分类、图像特征分类等）。

## 依赖环境

APScheduler==3.10.4

fastapi==0.115.12

numpy==1.23.5

pandas==2.2.3

scikit_learn==1.3.1

uvicorn==0.34.0

## 数据格式说明

### 1. 训练数据 (`train.jsonl`)

每行一个 JSON 对象，包含以下字段：

{
    "feature": [f1, f2, ..., fn],  # 特征向量（数值列表）
    "label": 0/1/...               # 分类标签（整数）
}


### 2. 测试数据 (`test.jsonl`)

每行一个 JSON 对象，仅需包含特征字段：

{
    "feature": [f1, f2, ..., fn]
}


## 使用步骤

### 1. 数据准备

* 将训练数据文件 `train.jsonl` 放置在 `./datasets/` 目录下。
* 将测试数据文件 `test.jsonl` 放置在 `./datasets/raw/` 目录下


### 2. 运行代码

python trainer.py 


### 3. 输出结果

* **预测结果文件** ：`./datasets/merge_test.jsonl`（包含标签的测试数据）。
* **训练模型** ：保存在 `./checkpoints/model.pkl`

### 4.部署服务

python main.py

### 附录：代码目录结构

.
├── datasets/
│   ├── train.jsonl
│   └── raw/
│       └── test.jsonl
├── checkpoints/
│   └── model.pkl
├── main.py

｜----- trainer.py
└── README.md
