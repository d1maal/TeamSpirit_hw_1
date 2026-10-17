from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from . import course_ovr as g, course_interactions as j
CATEGORY_NAMES = {
    'missed_decisive': 'Пропущена определяющая реплика',
    'late_false_reaction': 'Ложная реакция после определяющей',
    'wrong_sign_at_decisive': 'Неверный знак определяющей реакции',
    'false_reaction_in_neutral': 'Ложная реакция в neutral-диалоге',
    'correct': 'Верный итог',
}


def last_non_neutral(values):
    indices = np.flatnonzero(np.asarray(values) != 'neutral')
    return int(indices[-1]) if len(indices) else -1


def analyze(dialogs, predicted_turns):
    if len(dialogs) != len(predicted_turns):
        raise ValueError('Different dialogue counts')
    rows = []
    for record, pred in zip(dialogs.to_dict('records'), predicted_turns):
        truth = [r['reaction'] for r in record['labels']['user_turns'][1:]]
        pred = list(pred)
        if len(truth) != len(pred) or not set(pred) <= set(g.CLASSES):
            raise ValueError('Invalid turn predictions')
        gold_index = last_non_neutral(truth)
        pred_index = last_non_neutral(pred)
        gold = truth[gold_index] if gold_index >= 0 else 'neutral'
        if gold != record['reaction']:
            raise ValueError('Last-non-neutral rule disagrees with true dialogue target')
        predicted = pred[pred_index] if pred_index >= 0 else 'neutral'
        correct = gold == predicted
        missed = gold_index >= 0 and pred[gold_index] == 'neutral'
        wrong = gold_index >= 0 and pred[gold_index] not in ['neutral', truth[gold_index]]
        late = gold_index >= 0 and any((v != 'neutral' for v in pred[gold_index + 1:]))
        if correct:
            category = 'correct'
        elif gold_index < 0:
            category = 'false_reaction_in_neutral'
        elif pred_index > gold_index:
            category = 'late_false_reaction'
        elif pred_index == gold_index:
            category = 'wrong_sign_at_decisive'
        else:
            category = 'missed_decisive'
        user_texts = [t['text'] for t in record['turns'] if t['role'] == 'user'][1:]
        rows.append(dict(dialog_id=record['dialog_id'], truth=gold, prediction=predicted, correct=correct, category=category, missed_decisive=missed, late_false_reaction=late, wrong_sign_at_decisive=wrong, decisive_user=gold_index + 2 if gold_index >= 0 else None, selected_user=pred_index + 2 if pred_index >= 0 else None, decisive_text=user_texts[gold_index] if gold_index >= 0 else '', selected_text=user_texts[pred_index] if pred_index >= 0 else '', true_sequence=' → '.join(truth), predicted_sequence=' → '.join(pred)))
    cases = pd.DataFrame(rows)
    errors = cases[~cases.correct]
    counts = errors.category.value_counts().reindex([k for k in CATEGORY_NAMES if k != 'correct'], fill_value=0)
    table = counts.rename_axis('category').to_frame('errors')
    table['share_of_errors'] = table.errors / max(len(errors), 1)
    table['share_of_dialogues'] = table.errors / max(len(cases), 1)
    table.index = table.index.map(CATEGORY_NAMES)
    assert int(table.errors.sum()) == len(errors)
    flags = pd.DataFrame([dict(signal=CATEGORY_NAMES[name], all_dialogues=int(cases[name].sum()), erroneous_dialogues=int(errors[name].sum())) for name in ['missed_decisive', 'late_false_reaction', 'wrong_sign_at_decisive']]).set_index('signal')
    return dict(cases=cases, summary=table, overlapping_flags=flags, accuracy=float(cases.correct.mean()), n_errors=len(errors), n_dialogues=len(cases))


def reconstruct(dialogs, checkpoint_dir, n_splits=5):
    dialogs = dialogs.reset_index(drop=True)
    checkpoint_dir = Path(checkpoint_dir)
    signature = hashlib.sha256(Path(__file__).read_bytes() + Path(j.__file__).read_bytes() + Path(g.__file__).read_bytes() + dialogs[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records').encode()).hexdigest()
    cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)
    predicted = [None] * len(dialogs)
    available = []
    missing = []
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(cv.split(dialogs, dialogs.reaction, dialogs.group), 1):
            path = checkpoint_dir / f'G_signals_fold{fold}.joblib'
            if not path.exists():
                missing.append(fold)
                continue
            checkpoint = joblib.load(path)
            cache = checkpoint_dir / f'diagnostic_turns_{signature[:12]}_fold{fold}.joblib'
            cache_key = hashlib.sha256(path.read_bytes()).hexdigest()
            stored = joblib.load(cache) if cache.exists() else {}
            if stored.get('checkpoint_sha256') == cache_key:
                predictions = stored['turn_predictions']
            else:
                choice = checkpoint['choice']
                assert not set(dialogs.group.iloc[tr]) & set(dialogs.group.iloc[va])
                bundle = j.fit(dialogs.iloc[tr].reset_index(drop=True), 'G_signals', choice)
                valid = dialogs.iloc[va][['dialog_id', 'turns']].reset_index(drop=True)
                pairs = g.interactions(valid)
                x = bundle['encoder'].transform(pairs)
                scores = {label: model.decision_function(x) / scale for label, (model, scale) in bundle['models'].items()}
                ep = scores['positive'] - choice['thresholds'][0]
                en = scores['negative'] - choice['thresholds'][1]
                turn_pred = np.where(
                    (ep >= 0) | (en >= 0),
                    np.where(ep > en, 'positive', 'negative'),
                    'neutral',
                )
                actual = g.aggregate(
                    pairs,
                    scores['positive'],
                    scores['negative'],
                    len(valid),
                    choice['thresholds'],
                )
                if not np.array_equal(actual, np.asarray(checkpoint['pred'])):
                    raise ValueError(f'Fold {fold}: reconstructed predictions differ from checkpoint. Check data order, groups, code and sklearn version.')
                predictions = [turn_pred[pairs.dialog_pos.to_numpy() == d].tolist() for d in range(len(valid))]
                joblib.dump(dict(turn_predictions=predictions, checkpoint_sha256=cache_key), cache)
            for index, values in zip(va, predictions):
                predicted[index] = values
            available.extend(va.tolist())
    if not available:
        raise FileNotFoundError('No G_signals_fold*.joblib checkpoints in chosen directory')
    indices = sorted(available)
    result = analyze(dialogs.iloc[indices].reset_index(drop=True), [predicted[i] for i in indices])
    result['missing_folds'] = missing
    result['evaluated_indices'] = indices
    result['cases'].to_csv(checkpoint_dir / 'diagnostic_error_cases.csv', index=False)
    result['summary'].to_csv(checkpoint_dir / 'diagnostic_error_summary.csv')
    result['overlapping_flags'].to_csv(checkpoint_dir / 'diagnostic_overlapping_flags.csv')
    return result


def display_result(result):
    from IPython.display import display
    print(f"Диалогов: {result['n_dialogues']}; ошибок: {result['n_errors']}; accuracy={result['accuracy']:.4f}")
    if result.get('missing_folds'):
        print('НЕПОЛНАЯ оценка: отсутствуют фолды', result['missing_folds'])
    display(result['summary'].round(4))
    print('Дополнительные признаки могут пересекаться и встречаться при верном итоговом ответе:')
    display(result['overlapping_flags'])
    for code in CATEGORY_NAMES:
        if code == 'correct':
            continue
        examples = result['cases'].query('category == @code').head(3)
        if len(examples):
            print(CATEGORY_NAMES[code])
            display(examples[['dialog_id', 'truth', 'prediction', 'decisive_user', 'selected_user', 'decisive_text', 'selected_text', 'true_sequence', 'predicted_sequence']])
