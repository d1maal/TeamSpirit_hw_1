import re

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits


PATTERNS = [
    r"\b(?:not|no|never|cannot|can't|don't|doesn't|didn't|isn't|wasn't)\b",
    r"\b(?:wrong|incorrect|false|inaccurate|disagree|mistake|error)\b",
    r"\b(?:fix|correct|revise|rewrite|redo|change|instead|still)\b",
    r"\b(?:explain|clarify|elaborate|detail|mean|understand)\b",
    r"\b(?:thanks|thank you|grateful|appreciate)\b",
    r"\b(?:great|perfect|excellent|helpful|works|awesome)\b",
    r"\b(?:sorry|apologize|apologise|apologies)\b",
    r"\b(?:you(?:'re| are) right|my mistake|i was wrong|i made (?:a |an )?(?:mistake|error)|i misunderstood)\b",
    r"\b(?:you(?:'re| are) welcome|glad (?:to|it)|happy to help)\b",
]



PATTERNS = [re.compile(p) for p in PATTERNS]



TOKEN = re.compile(r'\b\w+\b')



class _DialogReactionModel:

    def __init__(self, base):
        self.components = {'base': base}

    def predict(self, dialogs: list[dict], batch_size=128) -> list[str]:
        return predict_models({'model': self}, dialogs, batch_size=batch_size)['model']



class Decision2xClassifier(_DialogReactionModel):
    pass



class ScoreStackingLRClassifier(_DialogReactionModel):

    def __init__(self, base, stack):
        super().__init__(base)
        self.components['stack'] = stack



def _custom_components(model):
    if isinstance(model, _DialogReactionModel):
        return model.components

    if isinstance(model, dict) and 'base' in model:
        return model
    return None



def _pairs(dialogs):
    rows = []
    for d, dialog in enumerate(dialogs):
        turns = dialog['turns']
        users = [i for i, t in enumerate(turns) if t['role'] == 'user']
        for j, i in enumerate(users[1:], 1):
            user = turns[i]['text']
            prev = turns[i - 1] if i else {}
            next_turn = turns[i + 1] if i + 1 < len(turns) else {}
            rows.append(dict(
                dialog_pos=d, user=user,
                assistant=prev.get('text', '') if prev.get('role') == 'assistant' else '',
                following=next_turn.get('text', '')[:600] if next_turn.get('role') == 'assistant' else '',
                position=j / max(len(users) - 1, 1), log_length=np.log1p(len(user)),
                n_users=len(users), questions=user.count('?'), exclamations=user.count('!')))
    return pd.DataFrame(rows)



def _signals(pairs):
    columns, words = [], {}
    for source in ('user', 'assistant', 'following'):
        texts = pairs[source].fillna('').str.lower().tolist()
        columns.extend([[float(bool(p.search(t))) for t in texts] for p in PATTERNS])
        words[source] = [set(TOKEN.findall(t)) for t in texts]
    for a, b in [('user', 'assistant'), ('user', 'following'), ('assistant', 'following')]:
        columns.append([len(x & y) / max(len(x | y), 1) for x, y in zip(words[a], words[b])])
    return sparse.csr_matrix(np.asarray(columns, dtype=np.float32).T)



def _scores(base, pairs, need_nb):
    blocks, nb_blocks = [], []
    for part in base['parts']:
        block = part['transformer'].transform(pairs[part['columns']]) * part['weight']
        block = sparse.csr_matrix(block)
        blocks.append(block)
        if part['name'] != 'structure':
            nb_blocks.append(block)
    signals = _signals(pairs)
    blocks.append(base['signal_scaler'].transform(signals) * base['signal_weight'])
    x = sparse.hstack(blocks, format='csr')
    scores = {name: model.decision_function(x) / scale
              for name, (model, scale) in base['models'].items()}
    nb_x = sparse.hstack(nb_blocks, format='csr') if need_nb else None
    return scores, signals, nb_x



def _decision(pairs, scores, thresholds, n):
    pos = scores['positive'] - thresholds[0]
    neg = scores['negative'] - thresholds[1]
    indices = np.flatnonzero((pos >= 0) | (neg >= 0))
    last = np.full(n, -1)
    np.maximum.at(last, pairs.dialog_pos.to_numpy(dtype=int)[indices], indices)
    result = np.full(n, 'neutral', dtype=object)
    found = last >= 0
    result[found] = np.where(pos[last[found]] > neg[last[found]], 'positive', 'negative')
    return result



def _stack_features(pairs, scores, signals, odds, thresholds, n):
    ids = pairs.dialog_pos.to_numpy(dtype=int)
    flags = signals.toarray()
    pos = scores['positive'] - thresholds[0]
    neg = scores['negative'] - thresholds[1]
    result = []
    for d in range(n):
        idx = np.flatnonzero(ids == d)
        if not len(idx):
            result.append([0.] * 12)
            continue
        margin = np.maximum(pos[idx], neg[idx])
        found = np.flatnonzero(margin >= 0)
        z = float(np.float32(odds[idx].max()))
        result.append([
            float(np.float32(scores['positive'][idx].max())),
            float(np.float32(scores['negative'][idx].max())),
            float(np.float32(margin.max())),
            float(np.float32(np.sort(margin)[-2:].mean())),
            margin[found[-1]] if len(found) else 0., float(bool(len(found))),
            -np.logaddexp(0., z), -np.logaddexp(0., -z), z,
            flags[idx, 4].max(), flags[idx, 1].max(), flags[idx, 24].max()
        ])
    return np.asarray(result)



def predict_models(models, dialogs, batch_size=128):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    result = {}
    groups = {}
    for name, model in models.items():
        components = _custom_components(model)
        if components is None:
            raise TypeError('predict_models принимает только кастомные модели реакции')
        else:
            groups.setdefault(id(components['base']), {})[name] = components
    for components in groups.values():
        result.update(_predict_custom_models(components, dialogs, batch_size))
    return {name: result[name] for name in models}



def _predict_custom_models(models, dialogs, batch_size):
    result = {name: [] for name in models}
    base = next(iter(models.values()))['base']
    need_nb = any('stack' in model for model in models.values())
    with threadpool_limits(limits=1):
        for start in range(0, len(dialogs), batch_size):
            batch = dialogs[start:start + batch_size]
            pairs = _pairs(batch)
            if pairs.empty:
                for name in models:
                    result[name].extend(['neutral'] * len(batch))
                continue
            scores, signals, nb_x = _scores(base, pairs, need_nb)
            for name, model in models.items():
                if 'stack' not in model:
                    pred = _decision(pairs, scores, base['thresholds'], len(batch))
                else:
                    stack = model['stack']
                    logp = stack['nb'].predict_log_proba(nb_x)
                    classes = list(stack['nb'].classes_)
                    odds = logp[:, classes.index(1)] - logp[:, classes.index(0)]
                    features = _stack_features(pairs, scores, signals, odds, base['thresholds'], len(batch))
                    pred = stack['meta'].predict(stack['scaler'].transform(features))
                    counts = np.bincount(pairs.dialog_pos.to_numpy(dtype=int), minlength=len(batch))
                    pred[counts == 0] = 'neutral'
                result[name].extend(map(str, pred))
    return result

