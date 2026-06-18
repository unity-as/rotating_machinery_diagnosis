# 旋转机械故障诊断系统 - 完整实现方案

## 硬件与技术栈
- **CPU**: R9 8945HX | **GPU**: RTX 5060 | **内存**: 16GB
- **GPU 加速**: CuPy（全源最短路径 APSP）
- **词袋模型**: gensim Doc2Vec（dm=0, PV-DBOW 模式）
- **配置管理**: OmegaConf + YAML
- **Web 前端**: Flask/FastAPI（后续可扩展 UDP 接口）

## 项目目录结构
```
rotating_machinery_diagnosis/
├── config/default.yaml           # 统一配置
├── main_pipeline.py              # 流水线入口
├── app.py                        # Web 服务入口
├── graph/
│   ├── visibility_graph.py       # 延迟加权可见性图（numba）
│   ├── line_graph.py             # 线图构建
│   └── word_generator.py         # NDD + TM 单词生成（MD5哈希）
├── models/
│   ├── pvdbow.py                 # gensim Doc2Vec 封装
│   └── classifier.py             # 全连接分类器
├── data/
│   ├── dataset.py                # 窗口采样
│   └── augment.py                # 数据增强
├── utils/
│   ├── gpu_apsp.py               # GPU 全源最短路径
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

## 训练策略：直推式学习（Transductive Learning）

### 重要：PV-DBOW 在全量数据上训练，划分在分类器阶段进行

本项目采用**直推式学习**策略，PV-DBOW 在所有文档（训练集+测试集）上训练，然后在分类器阶段才进行训练/测试划分。这是合理的，原因如下：

1. **PV-DBOW 的本质**：PV-DBOW 是无监督的图嵌入模型，学习的是图的结构表示，而非分类边界。它不使用标签信息，因此不存在传统意义上的"数据泄露"。

2. **直推式学习的优势**：
   - PV-DBOW 能够看到所有图的结构信息，学习到更完整的图表示空间
   - 在实际应用中，新数据到来时可以直接用已训练好的模型推断向量
   - 这与 Netpro2vec 原论文的使用方式一致

3. **分类器阶段的划分**：
   - 在 step4 中，对融合后的特征向量进行训练/测试划分
   - 分类器只在训练集上学习分类边界，在测试集上评估泛化能力
   - 这样可以真实评估分类器的性能

### 流程说明
```
阶段一：单词提取（全量）
    ↓
阶段二：PV-DBOW 训练（全量文档）
    ↓
阶段三：特征拼接（全量向量）
    ↓
阶段四：分类器训练（此处划分训练/测试集）
```

## 三阶段流水线

### 阶段一：单词提取（step1_extract_words.py）
- 对每个类别，逐窗口处理：截取振动信号 → 构建延迟加权可见性图 → 构建线图
- 对原图和线图分别计算 NDD + TM 单词
- 单词格式：
  - NDD: `ndd_{节点度}_{距离区间编号1}_{距离区间编号2}_...` → MD5 哈希
  - TM: `tm_{源节点度}_{目标节点度1}_{目标节点度2}_...` → MD5 哈希
- 单词按类别分开存放：`datas/words_G/{类别}/{index}.pkl`、`datas/words_LG/{类别}/{index}.pkl`
- 进度保存机制：每处理完一个窗口立即保存 pkl，恢复时删除最后一个文件并从该点重新生成
- 并行加速：多进程并行处理窗口，CuPy 独立 CUDA 上下文，处理完释放显存
- 完成 采样窗口数 批次后结束该阶段
- 同时生成语料文件 corpus_G.txt、corpus_LG.txt

### 阶段二：PV-DBOW 训练（step2_train_pvdbow.py + step3_fuse_and_save.py）
- 使用 gensim Doc2Vec(dm=0) 分别训练原图和线图的词袋模型
- **在全量文档上训练**（不划分训练/测试集）
- 训练完成后保存模型文件（pvdbow_G.model、pvdbow_LG.model）
- 将每个文档的单词序列转化为 512 维特征向量，保存为 vec_G.npy、vec_LG.npy
- 拼接原图和线图向量 → 1024 维融合向量，保存为 fused_features.npy
- 生成 t-SNE 图：原图向量分布、线图向量分布、融合向量分布

### 阶段三：分类器训练（step4_train_classifier.py）
- 加载融合向量
- **在此阶段划分训练集/测试集**（使用 stratified split 保持类别比例）
- 训练全连接分类器（可配置隐藏层、激活函数、BatchNorm、Dropout）
- 输出验证准确率、分类报告、混淆矩阵
- 生成 t-SNE 图：训练集 vs 验证集特征分布、训练曲线

## 图自身特征可视化（独立脚本）
- 提取每个窗口图的拓扑指标：平均节点度、最大距离、节点度分布、聚类系数等（13维或18维）
- 生成 t-SNE 图展示不同类别图的特征分布差异

## Web 前端
- 随机抽检振动信号，展示原始波形图
- 展示建图结果、单词统计、特征向量
- 展示分类诊断结果（含置信度）
- 统一配置修改接口（通过网页调整参数）
- 一键诊断：每类随机截取1个窗口，同时诊断并显示结果

## 后续扩展
- UDP 接口：接收物联网振动信号采集模块的实时数据
- 在线推理：实时故障检测

## 关于文件
- `datas/raw_datas/`：原始数据集目录，绝对不允许更改和删除，绝对不允许动里面的内容

# 安全重训指令

## 绝对规则（违反任意一条立刻中止）
1. 禁止删除 `datas/raw_datas/` 下的任何文件。
2. 每步完成后必须向我报告结果，并等我回复“下一步”才能继续。
3. 任何删除动作执行前，必须列出将要删除的文件清单（若文件数 >20 则只报数量+首尾文件名），得到我确认后方可删除。


## 减少算力浪费
1. 重训练前，了解有什么发生了改变，和什么需要重新训练。
2. 请保留旧单词文件，除非（除了窗口采样数的）单词生成相关参数有变化。
3. 在删除旧文件时，请务必说明你要删的是哪些文件（我要删掉所有旧单词、我要删掉netpro2vec模型）

---

## 步骤一：生成单词
1. 检查 `datas/words_G/` 和 `datas/words_LG/` 目录。
   - 如果两个目录下都有有效的单词文件，且从上次生成单词后 **没有修改过单词生成相关的 config 词条** （窗口采样数不算），则报告“单词文件已就绪且配置未变，无需重建”，直接跳到步骤二。
   - 否则，请和我说：不存在有效单词文件 或者 修改过 **修改过的单词生成相关的 config 词条** 词条 并执行下面的 2→3→4 进行单词重建。

2. 先清理语料库文件：
   - 列出 `datas/corpus_G.txt` 和 `datas/corpus_LG.txt` 文件是否存在，若存在，询问我是否需要删除语料库文件并向我确认。
   - 我确认后，删除 `datas/corpus_G.txt` 和 `datas/corpus_LG.txt`。

3. 清理旧的单词文件（先确认后删除）：
   - 列出 `datas/words_G/` 和 `datas/words_LG/` 目录是否存在，若存在，向我确认是否删除。
   - 我确认后清空这两个目录。

4. 启动单词生成脚本：
   python scripts/step1_extract_words.py
   验证：生成后检查两个目录下是否出现了新的单词文件，新单词文件数 > 0。

---

## 步骤二：重新训练词袋模型（从头训练）
1. 确认步骤一已通过，且 `datas/words_G/` 和 `datas/words_LG/` 下有单词文件。`datas/ds/label_order.txt` 存在。

2. 删除所有旧的模型和向量文件：
   - 列出需要删除的文件（如 `datas/ds/pvdbow_G.model`, `datas/ds/pvdbow_G.model.syn1neg.npy`, `datas/ds/pvdbow_G.model.wv.vectors.npy`, `datas/ds/vec_G.npy`
   `datas/ds/pvdbow_LG.model`, `datas/ds/pvdbow_LG.model.syn1neg.npy`, `datas/ds/pvdbow_LG.model.wv.vectors.npy`, `datas/ds/vec_LG.npy` 等），向我确认。
   - 我确认后，列出的文件全部删除（`raw_data` 不动）。

3. 启动词袋模型训练脚本：
   python scripts/step2_train_pvdbow.py
   验证：检查模型目录生成，且模型文件有效。

---

## 步骤三：生成并融合向量
1. 确认步骤二已完成。

2. 删除旧的融合向量文件：
  - 列出需要删除的文件 `datas/ds/fused_features.npy`，向我确认。
  - 我确认后，删除`datas/ds/fused_features.npy`（`raw_data` 不动）。

---

## 步骤四：重新训练分类器（从头训练）
1. 确认 `datas/ds/fused_features.npy` 存在。

2. 删除旧分类器模型（先列出确认）：
   - 列出要删除的模型文件（如 `classifier.pth`），向我确认后删除。

3. 训练分类器：
   python scripts/step4_train_classifier.py
   验证：输出准确率、F1 等指标，并保存新模型。

---

## 完成后输出清单
- 单词文件目录: …
- 词袋模型目录: …
- 融合向量路径: …
- 分类器模型路径: …
- 最终性能: accuracy=…, f1=…

## 测试脚本：
compare_outer_roll.py 关于outer和roll重合度高，专门比较
show_words_detail.py 展示单词的统计信息
check_train_test_acc.py 验证分类器在数据集内的准确率
debug_graph_features.py 生成基本图特征的tsne图
verify_accuracy.py 检查分类器的全集准确率