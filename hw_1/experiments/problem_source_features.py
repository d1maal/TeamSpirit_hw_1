import re
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
PATTERNS = {
    'technical_error': r'\b(?:traceback|stack trace|exception|(?:type|value|key|index|name|attribute|syntax|runtime|import|module.?not.?found|indentation|zero.?division)error|segmentation fault|syntax error|error (?:message|code)|exit code|compilation (?:error|failed)|failed to compile|doesn.t (?:run|compile)|won.t (?:run|compile))\b',
    'assistant_correction': r"\b(?:you (?:are|were|got it) (?:wrong|incorrect)|you(?:.ve| have)? (?:made a mistake|misunderstood|confused|missed|forgot|ignored)|your (?:answer|response|explanation|solution|code|calculation) (?:is|was|does|doesn.t|has|contains)|that(?:.s| is) (?:wrong|incorrect)|not (?:correct|true)|you said|you claimed)\b",
    'requirement_mismatch': r"\b(?:not what i (?:asked|requested|wanted|meant)|i (?:asked|requested|wanted|specified) (?:for|you|that)|you (?:didn.t|did not|haven.t|have not) (?:follow|include|answer|use|explain|provide)|instead of|i (?:said|told you)|doesn.t (?:match|meet|follow) (?:my|the)|without (?:changing|using))\b",
    'explicit_dissatisfaction': r"\b(?:unhelpful|useless|terrible|disappointed|frustrating|not helpful|not satisfied|waste of (?:my )?time|you.re not listening|stop (?:ignoring|repeating)|this is bad)\b",
    'assistant_apology': r'\b(?:sorry|apologize|apologise|apologies)\b',
}
NAMES = list(PATTERNS) + ['technical_and_apology', 'correction_and_apology', 'mismatch_and_apology', 'technical_without_explicit_complaint', 'technical_with_explicit_complaint']


class ProblemSourceFeatures(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(NAMES, dtype=object)

    def transform(self, x):
        columns = {}
        for name, pattern in PATTERNS.items():
            source = 'following' if name == 'assistant_apology' else 'user'
            regex = re.compile(pattern, re.I)
            columns[name] = np.array([bool(regex.search(text or '')) for text in x[source]], dtype=bool)
        technical = columns['technical_error']
        apology = columns['assistant_apology']
        complaint = columns['assistant_correction'] | columns['requirement_mismatch'] | columns['explicit_dissatisfaction']
        columns.update(
            technical_and_apology=technical & apology,
            correction_and_apology=columns['assistant_correction'] & apology,
            mismatch_and_apology=columns['requirement_mismatch'] & apology,
            technical_without_explicit_complaint=technical & ~complaint,
            technical_with_explicit_complaint=technical & complaint,
        )
        return sparse.csr_matrix(np.column_stack([columns[name] for name in NAMES]).astype(np.float32))
