from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import course_ovr as g, course_interactions as j, course_ablation as a, course_decision_weight as d
RESULT_DIR = Path('artifacts/course_positive_weight')
CONFIGS = [(0.1, 'balanced'), (0.3, 'balanced'), (1.0, 'balanced')]
VARIANTS = {'Decision_2x': 2.0, 'Positive_3x': 3.0, 'Positive_4x': 4.0, 'Positive_6x': 6.0}


def decisive_positive(p):
    mask = np.zeros(len(p), dtype=bool)
    y = p.reaction.to_numpy()
    ids = p.dialog_pos.to_numpy()
    for dialog in np.unique(ids):
        eligible = np.flatnonzero((ids == dialog) & (y != 'neutral'))
        if len(eligible) and y[eligible[-1]] == 'positive':
            mask[eligible[-1]] = True
    return mask


def mine(train, n_splits=2):
    p = g.interactions(train, True)
    margin = np.full(len(p), np.nan)
    for tr, va in d.splits(train, n_splits):
        pt = g.interactions(train.iloc[tr], True)
        pv = g.interactions(train.iloc[va][['dialog_id', 'turns']])
        _, xt, xv = next(j.matrix_variants(pt, pv, ['G_signals']))
        model, scale = d.fit_binary(
            xt,
            pt.reaction.to_numpy(),
            'positive',
            (0.3, 'balanced'),
            d.sample_weights(pt, (2.0, 2.0, 2.0)),
        )
        mask = p.dialog_pos.isin(va).to_numpy()
        assert mask.sum() == len(pv)
        margin[mask] = model.decision_function(xv) / scale
    assert np.isfinite(margin).all()
    return decisive_positive(p) & (margin < 0)


def weights(p, name, hard=None):
    base = d.sample_weights(p, (2.0, 2.0, 2.0))
    if name == 'Hard_positive_6x':
        if hard is None or len(hard) != len(p):
            raise ValueError('Training-only OOF hard mask required')
        base[hard] *= 3.0
    else:
        base[decisive_positive(p)] *= VARIANTS[name] / 2.0
    return base / base.mean()


def select(train, names, configs, inner_splits, cache_dir=None):
    pairs = g.interactions(train, True)
    positive = {name: np.full((len(configs), len(pairs)), np.nan) for name in names}
    negative = np.full((len(configs), len(pairs)), np.nan)
    for part, (tr, va) in enumerate(d.splits(train, inner_splits), 1):
        fit = train.iloc[tr].reset_index(drop=True)
        pt = g.interactions(fit, True)
        pv = g.interactions(train.iloc[va][['dialog_id', 'turns']])
        cache = Path(cache_dir) / f'inner{part}_models.joblib' if cache_dir is not None else None
        saved = joblib.load(cache) if cache is not None and cache.exists() else {}
        mask = pairs.dialog_pos.isin(va).to_numpy()
        assert mask.sum() == len(pv)
        if 'encoder' not in saved:
            enc = j.encoder('G_signals')
            xt = enc.fit_transform(pt)
            xv = enc.transform(pv)
            saved = {'encoder': enc, 'positive': {}, 'models': {}}
        else:
            enc = saved['encoder']
            xt = xv = None
        need_negative = 'negative' not in saved
        missing = [name for name in names if name not in saved['positive']]
        if need_negative or missing:
            if xt is None:
                xt = enc.transform(pt)
                xv = enc.transform(pv)
            y = pt.reaction.to_numpy()
            if need_negative:
                values = []
                models = []
                for config in configs:
                    model, scale = d.fit_binary(
                        xt,
                        y,
                        'negative',
                        config,
                        d.sample_weights(pt, (2.0, 2.0, 2.0)),
                    )
                    models.append((model, scale))
                    values.append(model.decision_function(xv) / scale)
                saved['negative'] = np.asarray(values)
                saved['negative_models'] = models
            for name in missing:
                hard = mine(fit) if name == 'Hard_positive_6x' else None
                w = weights(pt, name, hard)
                values = []
                models = []
                for config in configs:
                    model, scale = d.fit_binary(xt, y, 'positive', config, w)
                    models.append((model, scale))
                    values.append(model.decision_function(xv) / scale)
                saved['positive'][name] = np.asarray(values)
                saved['models'][name] = models
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(saved, cache)
        negative[:, mask] = saved['negative']
        for name in names:
            positive[name][:, mask] = saved['positive'][name]
    assert np.isfinite(negative).all() and all((np.isfinite(v).all() for v in positive.values()))
    baseline, search = j.choose_fast(
        train,
        pairs,
        {'positive': positive['Decision_2x'], 'negative': negative},
        configs,
    )
    choices = {'Decision_2x': baseline}
    searches = {'Decision_2x': search}
    kn = configs.index(baseline['negative'])
    tn = baseline['thresholds'][1]
    y = train.reaction.to_numpy()
    for name in names:
        if name == 'Decision_2x':
            continue
        best = None
        rows = []
        for k, config in enumerate(configs):
            for tp in g.THRESHOLDS:
                pred = g.aggregate(pairs, positive[name][k], negative[kn], len(train), (tp, tn))
                value = a.macro(y, pred)
                rows.append(dict(positive_config=k, positive_threshold=tp, macro_f1=value))
                if best is None or value > best['inner_macro_f1'] + 1e-12:
                    best = dict(
                        positive=config,
                        negative=baseline['negative'],
                        thresholds=(tp, tn),
                        inner_macro_f1=value,
                    )
        choices[name] = best
        searches[name] = pd.DataFrame(rows)
    return (choices, searches)


def outer(train, valid, choices, cache_dir=None):
    pt = g.interactions(train, True)
    pv = g.interactions(valid[['dialog_id', 'turns']])
    cache = Path(cache_dir) / 'outer_models.joblib' if cache_dir is not None else None
    saved = joblib.load(cache) if cache is not None and cache.exists() else {}
    base = choices['Decision_2x']
    y = pt.reaction.to_numpy()
    if not saved:
        enc = j.encoder('G_signals')
        xt = enc.fit_transform(pt)
        xv = enc.transform(pv)
        model, scale = d.fit_binary(
            xt,
            y,
            'negative',
            base['negative'],
            d.sample_weights(pt, (2.0, 2.0, 2.0)),
        )
        saved = {
            'encoder': enc,
            'negative_model': (model, scale),
            'negative_score': model.decision_function(xv) / scale,
            'outputs': {},
            'positive_models': {},
        }
    else:
        enc = saved['encoder']
        xt = xv = None
    negative = saved['negative_score']
    out = {}
    for name, choice in choices.items():
        if name in saved['outputs']:
            out[name] = saved['outputs'][name]
            continue
        if xt is None:
            xt = enc.transform(pt)
            xv = enc.transform(pv)
        hard = mine(train) if name == 'Hard_positive_6x' else None
        model, scale = d.fit_binary(xt, y, 'positive', choice['positive'], weights(pt, name, hard))
        positive = model.decision_function(xv) / scale
        pred = g.aggregate(pv, positive, negative, len(valid), choice['thresholds'])
        ep = positive - choice['thresholds'][0]
        en = negative - choice['thresholds'][1]
        turn = pd.DataFrame({'dialog_id': valid.dialog_id.to_numpy()[pv.dialog_pos.to_numpy()], 'user_number': pv.groupby('dialog_pos').cumcount().to_numpy() + 2, 'positive_score': positive, 'negative_score': negative, 'positive_threshold': choice['thresholds'][0], 'negative_threshold': choice['thresholds'][1], 'prediction': np.where((ep >= 0) | (en >= 0), np.where(ep > en, 'positive', 'negative'), 'neutral')})
        out[name] = dict(pred=pred, choice=choice, turn_scores=turn, hard_count=int(hard.sum()) if hard is not None else 0)
        saved['outputs'][name] = out[name]
        saved['positive_models'][name] = (model, scale)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(saved, cache)
    return out


def fingerprint(train, configs, outer, inner, hard_mining):
    source = Path(__file__).read_bytes() + b''.join((Path(m.__file__).read_bytes() for m in [g, j, a, d]))
    return hashlib.sha256(source + train[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records').encode() + repr((configs, outer, inner, hard_mining)).encode()).hexdigest()


def run(train, configs=CONFIGS, outer_splits=5, inner_splits=3, hard_mining=False):
    train = train.reset_index(drop=True)
    names = list(VARIANTS) + (['Hard_positive_6x'] if hard_mining else [])
    key = fingerprint(train, configs, outer_splits, inner_splits, hard_mining)
    dest = RESULT_DIR / key[:16]
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'results.joblib').exists():
        return joblib.load(dest / 'results.joblib')
    shared_key = fingerprint(train, configs, outer_splits, inner_splits, False)
    shared = RESULT_DIR / shared_key[:16] / 'shared_models'
    shared.mkdir(parents=True, exist_ok=True)
    preds = {name: np.full(len(train), '', dtype=object) for name in names}
    folds = []
    settings = []
    turns = []
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(d.splits(train, outer_splits), 1):
            cache = dest / f'fold{fold}.joblib'
            if cache.exists():
                data = joblib.load(cache)
            else:
                fit = train.iloc[tr].reset_index(drop=True)
                valid = train.iloc[va][['dialog_id', 'turns']].reset_index(drop=True)
                choices, searches = select(fit, names, configs, inner_splits, shared / f'fold{fold}')
                for name, search in searches.items():
                    search.to_csv(dest / f'{name}_fold{fold}_search.csv', index=False)
                data = outer(fit, valid, choices, shared / f'fold{fold}')
                joblib.dump(data, cache)
            for name, v in data.items():
                preds[name][va] = v['pred']
                value = a.macro(train.reaction.to_numpy()[va], v['pred'])
                folds.append(dict(variant=name, fold=fold, macro_f1=value))
                settings.append(dict(variant=name, fold=fold, **v['choice'], hard_count=v['hard_count']))
                turns.append(v['turn_scores'].assign(variant=name, fold=fold))
    assert all(((p != '').all() for p in preds.values()))
    from sklearn.metrics import precision_recall_fscore_support
    table = []
    ft = pd.DataFrame(folds)
    for name, pred in preds.items():
        precision, recall, f1, _ = precision_recall_fscore_support(
            train.reaction,
            pred,
            labels=['positive'],
            zero_division=0,
        )
        table.append(dict(variant=name, OOF_Macro_F1=a.macro(train.reaction.to_numpy(), pred), Mean_CV_Macro_F1=ft[ft.variant == name].macro_f1.mean(), positive_precision=precision[0], positive_recall=recall[0], positive_f1=f1[0]))
    table = pd.DataFrame(table).set_index('variant')
    table['Delta_vs_Decision_2x'] = table.OOF_Macro_F1 - table.loc['Decision_2x', 'OOF_Macro_F1']
    result = dict(
        table=table,
        predictions=preds,
        y=train.reaction.to_numpy(),
        fold_scores=ft,
        settings=settings,
        turn_scores=pd.concat(turns, ignore_index=True),
        dialog_ids=train.dialog_id.tolist(),
        fingerprint=key,
    )
    table.to_csv(dest / 'comparison.csv')
    result['turn_scores'].to_csv(dest / 'oof_turn_scores.csv', index=False)
    joblib.dump(result, dest / 'results.joblib')
    return result


def display_result(result):
    from IPython.display import display
    display(result['table'].round(4))
    display(pd.DataFrame(result['settings']))
    paired = result['fold_scores'].pivot(index='fold', columns='variant', values='macro_f1')
    display(paired.subtract(paired.Decision_2x, axis=0).round(4))
    for name, pred in result['predictions'].items():
        print(name)
        g.display_result(dict(y=result['y'], predictions=pred, macro_f1=result['table'].loc[name, 'OOF_Macro_F1'], mean_cv_macro_f1=result['table'].loc[name, 'Mean_CV_Macro_F1']))
