import numpy as np
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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
        seed=42
    )

    model.save(output_model_path)
    print(f"模型已保存: {output_model_path}")

    vectors = np.array([model.dv[f'g_{i}'] for i in range(len(documents))])
    return vectors


def load_pvdbow_model(model_path):
    return Doc2Vec.load(model_path)


def infer_vectors(model, corpus_file):
    documents = build_tagged_documents(corpus_file)
    vectors = np.array([model.infer_vector(doc.words) for doc in documents])
    return vectors
