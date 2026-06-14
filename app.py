import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import pickle
import json
import random
from flask import Flask, render_template, request, jsonify
from omegaconf import OmegaConf
from models.classifier import Classifier
from models.pvdbow import load_pvdbow_model, infer_vectors
from graph.visibility_graph import build_delayed_weighted_visibility_graph
from graph.line_graph import build_weighted_line_graph
from graph.word_generator import graph_to_words

app = Flask(__name__)

cfg = OmegaConf.load("config/default.yaml")
config_dict = OmegaConf.to_container(cfg, resolve=True)

CLASS_NAMES = ['normal', 'inner', 'outer', 'combine', 'roll']
CLASS_LABELS = {
    'normal': '正常',
    'inner': '内圈故障',
    'outer': '外圈故障',
    'combine': '复合故障',
    'roll': '滚动体故障'
}

_model_cache = {}
_model_timestamps = {}


def load_models():
    model_path = os.path.join(config_dict['output_dir'], config_dict['save_best_model'])
    model_G_path = os.path.join(config_dict['output_dir'], 'pvdbow_G.model')
    model_LG_path = os.path.join(config_dict['output_dir'], 'pvdbow_LG.model')

    current_ts = (
        os.path.getmtime(model_path) if os.path.exists(model_path) else 0,
        os.path.getmtime(model_G_path) if os.path.exists(model_G_path) else 0,
        os.path.getmtime(model_LG_path) if os.path.exists(model_LG_path) else 0,
    )

    if 'classifier' in _model_cache and _model_timestamps.get('ts') == current_ts:
        return _model_cache['classifier'], _model_cache['dv_G'], _model_cache['dv_LG'], _model_cache['device']

    device = torch.device(config_dict['device'] if torch.cuda.is_available() else 'cpu')
    fused_path = os.path.join(config_dict['output_dir'], 'fused_features.npy')
    if not os.path.exists(fused_path):
        return None, None, None, device
    X = np.load(fused_path)
    input_dim = X.shape[1]
    model = Classifier(input_dim, len(CLASS_NAMES), config_dict).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    dv_G = load_pvdbow_model(model_G_path) if os.path.exists(model_G_path) else None
    dv_LG = load_pvdbow_model(model_LG_path) if os.path.exists(model_LG_path) else None
    _model_cache.update({'classifier': model, 'dv_G': dv_G, 'dv_LG': dv_LG, 'device': device})
    _model_timestamps['ts'] = current_ts
    return model, dv_G, dv_LG, device


def _diagnose_signal(signal_np):
    model, dv_G, dv_LG, device = load_models()
    if model is None or dv_G is None or dv_LG is None:
        return {'error': '模型未训练'}
    G = build_delayed_weighted_visibility_graph(signal_np, config_dict['tau'])
    LG = build_weighted_line_graph(G)
    wG = graph_to_words(G, config_dict)
    wLG = graph_to_words(LG, config_dict)
    vec_G = dv_G.infer_vector(wG, epochs=30).reshape(1, -1)
    vec_LG = dv_LG.infer_vector(wLG, epochs=30).reshape(1, -1)
    fused = np.concatenate([vec_G, vec_LG], axis=1).astype(np.float32)
    with torch.no_grad():
        x = torch.tensor(fused, dtype=torch.float32).to(device)
        output = model(x)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return {
        'prediction': CLASS_NAMES[pred_idx],
        'label': CLASS_LABELS[CLASS_NAMES[pred_idx]],
        'confidence': float(probs[pred_idx]),
        'probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        'graph_nodes': G.number_of_nodes(),
        'graph_edges': G.number_of_edges(),
        'line_graph_nodes': LG.number_of_nodes(),
        'line_graph_edges': LG.number_of_edges(),
        'words_G_count': len(wG),
        'words_LG_count': len(wLG)
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/random_sample')
def random_sample():
    raw_dir = config_dict['raw_data_dir']
    samples = []
    for cls in CLASS_NAMES:
        for fname in os.listdir(raw_dir):
            if cls in fname.lower() and fname.endswith('.npy'):
                sig = np.load(os.path.join(raw_dir, fname), mmap_mode='r')
                win = config_dict['window_size']
                if len(sig) >= win:
                    start = random.randint(0, len(sig) - win)
                    window = sig[start:start+win].tolist()
                    samples.append({
                        'class': cls,
                        'label': CLASS_LABELS[cls],
                        'signal': window,
                        'file': fname
                    })
                break
    return jsonify(samples)


@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    data = request.json
    signal = np.array(data['signal'], dtype=np.float64)
    result = _diagnose_signal(signal)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/batch_diagnose')
def batch_diagnose():
    raw_dir = config_dict['raw_data_dir']
    win = config_dict['window_size']
    results = []
    for cls in CLASS_NAMES:
        for fname in os.listdir(raw_dir):
            if cls in fname.lower() and fname.endswith('.npy'):
                sig = np.load(os.path.join(raw_dir, fname), mmap_mode='r')
                if len(sig) >= win:
                    start = random.randint(0, len(sig) - win)
                    window = sig[start:start+win]
                    signal_list = window.tolist()
                    diag = _diagnose_signal(window)
                    diag['expected'] = cls
                    diag['expected_label'] = CLASS_LABELS[cls]
                    diag['signal'] = signal_list
                    diag['file'] = fname
                    diag['correct'] = diag['prediction'] == cls
                    results.append(diag)
                break
    return jsonify(results)


@app.route('/api/model_info')
def model_info():
    info = {
        'config': {
            'window_size': config_dict['window_size'],
            'tau': config_dict['tau'],
            'vector_size': config_dict['vector_size'],
            'pvdbow_epochs': config_dict['pvdbow_epochs'],
            'pvdbow_learning_rate': config_dict['pvdbow_learning_rate'],
            'pvdbow_min_count': config_dict['pvdbow_min_count'],
            'pvdbow_down_sampling': config_dict['pvdbow_down_sampling'],
            'classifier_hidden_dims': config_dict['classifier_hidden_dims'],
            'classifier_activation': config_dict['classifier_activation'],
            'classifier_use_bn': config_dict['classifier_use_bn'],
            'classifier_batch_size': config_dict['classifier_batch_size'],
            'classifier_epochs': config_dict['classifier_epochs'],
            'classifier_lr': config_dict['classifier_lr'],
            'dropout_rate': config_dict['dropout_rate'],
            'weight_decay': config_dict['weight_decay'],
            'early_stop_patience': config_dict['early_stop_patience'],
            'ndd_delta': config_dict['ndd_delta'],
            'ndd_thresholds': config_dict['ndd_thresholds'],
            'tm_s_steps': config_dict['tm_s_steps'],
            'tm_thresholds': config_dict['tm_thresholds'],
            'tm_top_k': config_dict['tm_top_k'],
            'max_segments_per_file': config_dict['max_segments_per_file'],
            'test_ratio': config_dict['test_ratio'],
            'use_augmentation': config_dict['use_augmentation'],
        },
        'classes': CLASS_NAMES,
        'class_labels': CLASS_LABELS,
        'device': str(torch.device(config_dict['device'] if torch.cuda.is_available() else 'cpu'))
    }
    fused_path = os.path.join(config_dict['output_dir'], 'fused_features.npy')
    if os.path.exists(fused_path):
        X = np.load(fused_path)
        info['feature_dim'] = int(X.shape[1])
        info['total_samples'] = int(X.shape[0])
    best_model_path = os.path.join(config_dict['output_dir'], config_dict['save_best_model'])
    if os.path.exists(best_model_path):
        info['model_exists'] = True
    return jsonify(info)


@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(config_dict)


@app.route('/api/config', methods=['POST'])
def update_config():
    global config_dict
    updates = request.json
    for key, value in updates.items():
        if key in config_dict:
            config_dict[key] = value
    return jsonify({'status': 'ok', 'config': config_dict})


@app.route('/api/reload_models', methods=['POST'])
def reload_models():
    _model_cache.clear()
    _model_timestamps.clear()
    model, dv_G, dv_LG, device = load_models()
    if model is None:
        return jsonify({'error': '模型文件不存在'}), 400
    return jsonify({'status': 'ok', 'message': '模型已重新加载'})


if __name__ == '__main__':
    app.run(host=config_dict.get('web_host', '0.0.0.0'),
            port=config_dict.get('web_port', 5000),
            debug=config_dict.get('debug_mode', False))
