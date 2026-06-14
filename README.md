# 旋转机械故障诊断系统

基于 Netpro2vec 图嵌入的旋转机械振动信号智能诊断系统。

## 描述：
此代码是专利：基于延迟加权可视图和Netpro2vec的旋转机械故障诊断方法 的五元分类实现。
仅供学习参考和测试，用于商业用途而产生的后果概不负责。

## 实现硬件

- **CPU**: R9 8945HX
- **GPU**: RTX 5060，cuda 12.8
- **内存**: 16GB

## 项目结构

```
rotating_machinery_diagnosis/
├── config/default.yaml           # 统一配置
├── main_pipeline.py              # 流水线入口
├── app.py                        # Web 服务入口
├── graph/
│   ├── visibility_graph.py       # 延迟加权可见性图（numba 加速）
│   ├── line_graph.py             # 线图构建
│   └── word_generator.py         # NDD + TM 单词生成（MD5 哈希）
├── models/
│   ├── pvdbow.py                 # gensim Doc2Vec 封装
│   └── classifier.py             # 全连接分类器
├── data/
│   ├── dataset.py                # 窗口采样
│   └── augment.py                # 数据增强
├── utils/
│   ├── gpu_apsp.py               # GPU 全源最短路径（CuPy）
│   ├── visualizer.py             # t-SNE + 训练曲线
│   └── io_utils.py               # 缓存读写
├── scripts/
│   ├── step1_extract_words.py    # 单词提取
│   ├── step2_train_pvdbow.py     # PV-DBOW 训练
│   ├── step3_fuse_and_save.py    # 特征拼接
│   └── step4_train_classifier.py # 分类器训练
├── templates/                    # Web 前端页面
└── static/                       # 静态资源
```

## 快速开始

### 1. 安装依赖

```bash
# 安装依赖
pip install -r requirements.txt

# 下载 Cupy安装依赖 根据cuda版本确认。如12.8下载cupy-cuda12x和torch的cu128版本。
pip install cupy-cuda12x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. 准备数据

将振动信号数据（.npy 格式）放入 `datas/raw_datas/` 目录。
.npy 文件要包含关键字
`normal`
`inner`
`outer`
`combine`
`roll`

### 3. 运行流水线

```bash
python main_pipeline.py
```

或分步运行：

```bash
# 步骤一：提取单词
python scripts/step1_extract_words.py

# 步骤二：训练 PV-DBOW
python scripts/step2_train_pvdbow.py

# 步骤三：特征融合
python scripts/step3_fuse_and_save.py

# 步骤四：训练分类器
python scripts/step4_train_classifier.py
```

### 4. 启动 Web 服务

```bash
python app.py
```

访问 http://127.0.0.1:5000

## 算法流程

### 阶段一：单词提取 step1

1. 随机截取振动信号窗口（默认 800 点）
2. 构建延迟加权可见性图（原图）
3. 构建线图
4. 计算 NDD（节点距离分布）单词
5. 计算 TM（转移矩阵）单词
6. MD5 哈希压缩单词

### 阶段二：PV-DBOW 训练 step2-3

1. 使用 gensim Doc2Vec（dm=0）分别训练原图和线图的词袋模型
2. 提取文档向量（512 维）
3. 拼接原图和线图向量 → 1024 维融合向量

### 阶段三：分类器训练 step4

1. 划分训练集/测试集
2. 训练全连接分类器
3. 输出准确率、分类报告、混淆矩阵

### 补充说明：
修改配置文件后，请先删除需要被覆盖的旧数据。

## 诊断类别

| 类别 | 说明 |
|------|------|
| normal | 正常状态 |
| inner | 内圈故障 |
| outer | 外圈故障 |
| combine | 复合故障 |
| roll | 滚动体故障 |

## 配置说明

主要配置参数在 `config/default.yaml`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| window_size | 800 | 窗口大小 |
| tau | 2 | 延迟参数 |
| vector_size | 512 | 嵌入维度 |
| pvdbow_epochs | 30 | PV-DBOW 训练轮数 |
| classifier_epochs | 150 | 分类器训练轮数 |
| test_ratio | 0.2 | 测试集比例 |
| num_workers | 4 | 并行进程数 |

## 技术栈

- **GPU 加速**: CuPy（全源最短路径 APSP）
- **图构建**: NetworkX + Numba
- **词袋模型**: gensim Doc2Vec（PV-DBOW）
- **分类器**: PyTorch 全连接网络
- **Web 前端**: Flask + Chart.js
- **配置管理**: OmegaConf + YAML

## Web 功能

- 随机抽检振动信号
- 一键诊断（每类随机截取）
- 在线故障诊断
- 概率分布可视化
- 系统配置修改

## 测试参数与性能指标

### 不同样本数/类 + 不同节点标识版本对比

| 节点标识 | 样本数/类 | 真实准确率 | 训练/测试集准确率 |
|----------|-----------|-----------|-----------------|
| 节点索引 (0,1,2,...) | 1000 | 31.2% | 100.00% |
| 节点度数 | 1000 | 51.2% | 100.00% |
| 节点度数 | 2500 | **78.8%** | 100.00% |

### 1000 样本/类 节点索引版（31.2%）

| 类别 | 正确 | 总数 | 准确率 |
|------|------|------|--------|
| 正常 | 50 | 50 | 100.0% |
| 内圈故障 | 0 | 50 | 0.0% |
| 外圈故障 | 1 | 50 | 2.0% |
| 复合故障 | 0 | 50 | 0.0% |
| 滚动体故障 | 27 | 50 | 54.0% |
| **总计** | **78** | **250** | **31.2%** |

### 1000 样本/类 节点度数版（51.2%）

| 类别 | 正确 | 总数 | 准确率 |
|------|------|------|--------|
| 正常 | 50 | 50 | 100.0% |
| 内圈故障 | 0 | 50 | 0.0% |
| 外圈故障 | 14 | 50 | 28.0% |
| 复合故障 | 0 | 50 | 0.0% |
| 滚动体故障 | 27 | 50 | 54.0% |
| **总计** | **78** | **250** | **51.2%** |

### 2500 样本/类 节点度数版（78.8%）（当前测试版本）

| 类别 | 正确 | 总数 | 准确率 |
|------|------|------|--------|
| 正常 | 49 | 50 | 98.0% |
| 内圈故障 | 36 | 50 | 72.0% |
| 外圈故障 | 15 | 50 | 30.0% |
| 复合故障 | 49 | 50 | 98.0% |
| 滚动体故障 | 48 | 50 | 96.0% |
| **总计** | **197** | **250** | **78.8%** |

### 当前版本的混淆矩阵（2500 样本/类）

```
              normal  inner  outer  combine  roll
    normal      49      0      0       0      1
     inner       0     36      0      14      0
     outer       0      0     15       0     35
   combine       0      1      0      49      0
      roll       1      0      1       0     48
```

### 准确率分析

- **训练/测试集 100%**：PV-DBOW 在全量文档上训练后，特征几乎线性可分，所有训练样本完美预测
- **节点索引 vs 节点度数**：基于时间排序的节点索引（0,1,2,...）实际准确率更低（31.2% vs 51.2%），换用节点度数捕捉到了更稳定的结构特征

- **样本数影响**：采样节点度后，1000→2500 样本/类，真实准确率从 51.2% 提升至 78.8%，更多样本有助于 PV-DBOW 学习更稳定的文档表示
- **过拟合问题**：PV-DBOW 在全量文档上训练，训练集/测试集 100% 准确率是因文档向量直接从嵌入层获取；但样本不足时模型泛化能力差，`infer_vector()` 对新文档推断质量低，导致真实准确率仅 78.8%，因此仅需增大训练规模，即可提升准确率
- **主要混淆**：当前模型下，外圈故障被误判为滚动体故障，内圈故障被误判为复合故障

## 后续扩展

- UDP 接口：接收物联网振动信号采集模块
- 在线推理：实时故障检测

## 贡献者
伟大的：
- [unity-as](https://github.com/unity-as)