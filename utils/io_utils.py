import os
import pickle


def get_cached_ids(words_dir, class_name=None):
    ids = set()
    if not os.path.exists(words_dir):
        return ids
    if class_name is None:
        for sub in os.listdir(words_dir):
            sub_path = os.path.join(words_dir, sub)
            if os.path.isdir(sub_path):
                ids.update(get_cached_ids(sub_path))
                for fname in os.listdir(sub_path):
                    if fname.endswith('.pkl'):
                        try:
                            ids.add(int(fname.replace('.pkl', '')))
                        except:
                            pass
    else:
        class_dir = os.path.join(words_dir, class_name)
        if os.path.exists(class_dir):
            for fname in os.listdir(class_dir):
                if fname.endswith('.pkl'):
                    try:
                        ids.add(int(fname.replace('.pkl', '')))
                    except:
                        pass
    return ids


def save_words(words, class_name, idx, words_dir):
    class_dir = os.path.join(words_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    with open(os.path.join(class_dir, f'{idx}.pkl'), 'wb') as f:
        pickle.dump(words, f)


def load_words(class_name, idx, words_dir):
    pkl_path = os.path.join(words_dir, class_name, f'{idx}.pkl')
    if not os.path.exists(pkl_path):
        return []
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def delete_last_cached(words_dir, class_name, last_idx):
    pkl_path = os.path.join(words_dir, class_name, f'{last_idx}.pkl')
    if os.path.exists(pkl_path):
        os.remove(pkl_path)


def validate_and_clean_cached(words_dir, class_map):
    import pickle as pkl
    good_ids = set()
    for cls in class_map:
        cls_dir = os.path.join(words_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.endswith('.pkl'):
                continue
            fpath = os.path.join(cls_dir, fname)
            try:
                with open(fpath, 'rb') as f:
                    pkl.load(f)
                idx = int(fname.replace('.pkl', ''))
                good_ids.add((cls, idx))
            except (EOFError, pkl.UnpicklingError, ValueError):
                os.remove(fpath)
    return good_ids
