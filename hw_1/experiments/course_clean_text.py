from pathlib import Path
import hashlib
import re
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from . import course_ovr as g, course_interactions as j, course_ablation as a
RESULT_DIR = Path('artifacts/course_clean_text')
CONFIGS = [(0.1, 'balanced'), (0.3, 'balanced'), (1.0, 'balanced')]
NAMES = ['G_signals', 'G_raw_clean']


def clean_user(text):
    text = re.sub(r'<blockquote\b[^>]*>.*?</blockquote>', ' ', text, flags=re.I | re.S)
    lines = []
    fence = None
    trace = False
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r'^(`{3,}|~{3,})', stripped)
        if match:
            token = match.group(1)[0]
            if fence == token:
                fence = None
            elif fence is None:
                fence = token
            continue
        if fence is not None:
            continue
        if re.match(r'^>', stripped):
            continue
        if re.match(r'^Traceback \(most recent call last\):', stripped):
            trace = True
            continue
        if re.match(r'^(?:[\w.]+(?:Error|Exception)|Error|Exception)(?::|\b)', stripped):
            trace = False
            continue
        if trace and (not stripped or line.startswith((' ', '\t')) or stripped.startswith(('File ', '^'))):
            continue
        if trace:
            trace = False
        if re.match(r'^\s*File ["\'].*["\'], line \d+', line):
            continue
        if re.match(r'^\s*(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|\[?(?:DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL)\]?\s*[:\-])', line):
            continue
        if re.match(r'^\s*(?:import \w|from [\w.]+ import |(?:async )?def \w+\(|class \w+[:(]|(?:print|console\.log)\(|[\w.\[\]"\']+\s*(?:=|:=)\s*[^=]|[{};]+\s*$)', line):
            continue
        line = re.sub(r'`+[^`\n]*`+', ' ', line)
        lines.append(line)
    return '\n'.join(lines).strip()


def cleaned(p):
    return [clean_user(text) or '__empty_clean_text__' for text in p.user]


def matrices(pt, pv):
    base = j.encoder('G_signals')
    xt = base.fit_transform(pt)
    xv = base.transform(pv)
    yield ('G_signals', xt, xv)
    clean = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=12000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    ct = clean.fit_transform(cleaned(pt))
    cv = clean.transform(cleaned(pv))
    yield ('G_raw_clean', sparse.hstack([xt, ct], format='csr'), sparse.hstack([xv, cv], format='csr'))


def inner(train, configs, n_splits, n_jobs):
    p = g.interactions(train, True)
    scores = {name: {label: np.full((len(configs), len(p)), np.nan) for label in ['positive', 'negative']} for name in NAMES}
    for tr, va in StratifiedGroupKFold(n_splits, shuffle=True, random_state=42).split(train, train.reaction, train.group):
        assert not set(train.group.iloc[tr]) & set(train.group.iloc[va])
        pt = g.interactions(train.iloc[tr], True)
        pv = g.interactions(train.iloc[va].drop(columns=['labels', 'reaction']))
        mask = p.dialog_pos.isin(va).to_numpy()
        assert mask.sum() == len(pv)
        for name, xt, xv in matrices(pt, pv):
            for label, k, values in j.parallel_scores(xt, xv, pt.reaction.to_numpy(), configs, n_jobs):
                scores[name][label][k, mask] = values
    selections = {}
    searches = {}
    for name in NAMES:
        assert all((np.isfinite(v).all() for v in scores[name].values()))
        selections[name], searches[name] = j.choose_fast(train, p, scores[name], configs)
    return (selections, searches)


def fingerprint(train, configs, outer, inner):
    source = Path(__file__).read_bytes() + b''.join((Path(m.__file__).read_bytes() for m in [g, j, a]))
    return hashlib.sha256(source + train[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records').encode() + repr((configs, outer, inner)).encode()).hexdigest()


def run(train, configs=CONFIGS, outer_splits=5, inner_splits=3, n_jobs=2):
    train = train.reset_index(drop=True)
    key = fingerprint(train, configs, outer_splits, inner_splits)
    dest = RESULT_DIR / key[:16]
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'results.joblib').exists():
        return joblib.load(dest / 'results.joblib')
    predictions = {name: np.full(len(train), '', dtype=object) for name in NAMES}
    folds = []
    settings = []
    turns = []
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(StratifiedGroupKFold(outer_splits, shuffle=True, random_state=42).split(train, train.reaction, train.group), 1):
            assert not set(train.group.iloc[tr]) & set(train.group.iloc[va])
            cache = dest / f'fold{fold}.joblib'
            if cache.exists():
                data = joblib.load(cache)
            else:
                fit = train.iloc[tr].reset_index(drop=True)
                valid = train.iloc[va][['dialog_id', 'turns']].reset_index(drop=True)
                selections, searches = inner(fit, configs, inner_splits, n_jobs)
                for name, search in searches.items():
                    search.to_csv(dest / f'{name}_fold{fold}_search.csv', index=False)
                pt = g.interactions(fit, True)
                pv = g.interactions(valid)
                data = {}
                for name, xt, xv in matrices(pt, pv):
                    choice = selections[name]
                    scores = {}
                    for label in ['positive', 'negative']:
                        model, scale = g.fit_binary(xt, pt.reaction.to_numpy(), label, choice[label])
                        scores[label] = model.decision_function(xv) / scale
                    pred = g.aggregate(
                        pv,
                        scores['positive'],
                        scores['negative'],
                        len(valid),
                        choice['thresholds'],
                    )
                    ep = scores['positive'] - choice['thresholds'][0]
                    en = scores['negative'] - choice['thresholds'][1]
                    table = pd.DataFrame({'dialog_id': valid.dialog_id.to_numpy()[pv.dialog_pos.to_numpy()], 'user_number': pv.groupby('dialog_pos').cumcount().to_numpy() + 2, 'positive_score': scores['positive'], 'negative_score': scores['negative'], 'positive_threshold': choice['thresholds'][0], 'negative_threshold': choice['thresholds'][1], 'prediction': np.where((ep >= 0) | (en >= 0), np.where(ep > en, 'positive', 'negative'), 'neutral')})
                    data[name] = dict(pred=pred, choice=choice, turn_scores=table)
                joblib.dump(data, cache)
            for name, values in data.items():
                predictions[name][va] = values['pred']
                score = a.macro(train.reaction.to_numpy()[va], values['pred'])
                folds.append(dict(variant=name, fold=fold, macro_f1=score))
                settings.append(dict(variant=name, fold=fold, **values['choice']))
                turns.append(values['turn_scores'].assign(variant=name, fold=fold))
    assert all(((v != '').all() for v in predictions.values()))
    ft = pd.DataFrame(folds)
    table = pd.DataFrame([dict(variant=name, OOF_Macro_F1=a.macro(train.reaction.to_numpy(), pred), Mean_CV_Macro_F1=ft[ft.variant == name].macro_f1.mean()) for name, pred in predictions.items()]).set_index('variant')
    table['Delta_vs_G'] = table.OOF_Macro_F1 - table.loc['G_signals', 'OOF_Macro_F1']
    result = dict(
        table=table,
        fold_scores=ft,
        settings=settings,
        predictions=predictions,
        y=train.reaction.to_numpy(),
        dialog_ids=train.dialog_id.tolist(),
        turn_scores=pd.concat(turns, ignore_index=True),
        fingerprint=key,
    )
    result['turn_scores'].to_csv(dest / 'oof_turn_scores.csv', index=False)
    table.to_csv(dest / 'comparison.csv')
    joblib.dump(result, dest / 'results.joblib')
    return result


def display_result(result):
    from IPython.display import display
    display(result['table'].round(4))
    paired = result['fold_scores'].pivot(index='fold', columns='variant', values='macro_f1')
    paired['clean_minus_control'] = paired.G_raw_clean - paired.G_signals
    display(paired.round(4))
    for name, pred in result['predictions'].items():
        print(name)
        g.display_result(dict(y=result['y'], predictions=pred, macro_f1=result['table'].loc[name, 'OOF_Macro_F1'], mean_cv_macro_f1=result['table'].loc[name, 'Mean_CV_Macro_F1']))
