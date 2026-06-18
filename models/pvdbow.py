import numpy as np
import os
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec


class EpochProgressCallback(CallbackAny2Vec):
    def __init__(self, model_name):
        self.model_name = model_name
        self.start_time = None
        self.epoch_count = 0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"开始训练 {self.model_name} ...")

    def on_epoch_end(self, model, logs=None):
        self.epoch_count += 1
        elapsed = time.time() - self.start_time
        print(f"  {self.model_name} Epoch {self.epoch_count}/30, 已用时 {elapsed:.0f}s")

    def on_train_end(self, logs=None):
        elapsed = time.time() - self.start_time
        print(f"  {self.model_name} 训练完成, 总耗时 {elapsed:.0f}s")


def build_tagged_documents(corpus_file):
    documents = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            words = line.strip().split()
            if words:
                documents.append(TaggedDocument(words=words, tags=[f'g_{idx}']))
    return documents


def train_pvdbow(corpus_file, output_model_path, cfg):
    vector_size = cfg.get('vector_size', 512)
    epochs = cfg.get('pvdbow_epochs', 30)
    min_count = cfg.get('pvdbow_min_count', 5)
    down_sampling = cfg.get('pvdbow_down_sampling', 0.0001)
    workers = cfg.get('pvdbow_workers', 4)
    learning_rate = cfg.get('pvdbow_learning_rate', 0.025)

    print(f"构建文档集合: {corpus_file}")
    documents = build_tagged_documents(corpus_file)
    print(f"文档数: {len(documents)}")

    print(f"训练 PV-DBOW (vector_size={vector_size}, epochs={epochs}, min_count={min_count}) ...")
    model = Doc2Vec(
        documents,
        vector_size=vector_size,
        window=0,
        min_count=min_count,
        dm=0,
        sample=down_sampling,
        workers=workers,
        epochs=epochs,
        alpha=learning_rate,
        seed=42,
        callbacks=[EpochProgressCallback(output_model_path.split('/')[-1])]
    )

    model.save(output_model_path)
    print(f"模型已保存: {output_model_path}")

    vectors = np.array([model.infer_vector(doc.words, epochs=30) for doc in documents])
    return vectors


def load_pvdbow_model(model_path):
    return Doc2Vec.load(model_path)


def infer_vectors(model, corpus_file):
    documents = build_tagged_documents(corpus_file)
    vectors = np.array([model.infer_vector(doc.words) for doc in documents])
    return vectors
