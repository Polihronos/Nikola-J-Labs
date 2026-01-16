# MultiSourceBayesianBeliefUpdater — Technical Development Description

## Purpose

MultiSourceBayesianBeliefUpdater is a generic discrete Bayesian filtering component that maintains and **updates a posterior** belief distribution over a fixed set of discrete hypotheses (e.g., ["Risk-On", "Neutral", "Risk-Off"]) given multiple evidence streams (macro, sentiment, volatility, flows, etc.).

It is designed for JT products where signals arrive from heterogeneous (diversified) sources that may be:

 - sampled at different frequencies,
 - not correlated with each other,
 - sometimes missing (exeption),
 - differently reliable over time.

The updater supports:

 - multi-source evidence fusion via weighted likelihood combination,
 - sequential belief updating (posterior at t becomes prior at t+1),
 - forgetting / inertia to stabilize beliefs and avoid overreacting,
 - weakening detection to flag sharp posterior deterioration for any hypothesis.

## Conceptual Model

**State**

At any time t, the model tracks:

 - prior_t: belief distribution before ingesting evidence at time t
 - posterior_t: belief distribution after ingesting evidence at time t
 - history: stored posterior snapshots for time-series reasoning

Beliefs are a categorical distribution over hypotheses:

    P(Hi​);     i = 1..n;       ∑​P(Hi​)=1

**Evidence**

At each time step the updater receives a dict:

```python
        evidence_by_source = {
        "macro":     [score_H1, score_H2, ..., score_Hn],
        "sentiment": [score_H1, score_H2, ..., score_Hn],
        "vol":       [score_H1, score_H2, ..., score_Hn],
        }
```

Each source provides a per-hypothesis score vector. Scores are converted into a likelihood distribution:

$$
P(Ds​∣Hi​)
$$

using a temperature-softmax.

Important:
The component treats each source as an independent evidence channel unless you explicitly model dependencies elsewhere. This is the correct Bayesian fusion assumption for “streams that are not correlated” (independent conditional on the hypothesis).

**Evidence Fusion**

Likelihood conversion
Per source, scores are transformed into a normalized likelihood distribution:

$$
Ls​(i)=softmax(scores_{i} / T​​)
$$

where: T = temperature.
Higher T flattens likelihoods (less confident evidence). Lower T sharpens them.

**Weighted multi-source combination**

To combine multiple sources without requiring correlation, the updater uses product-of-experts style fusion under conditional independence:

$$
logL(i)=\sum_{s∈S}^{i} ​ws​⋅log(L_{s}​(i)+ϵ)
$$

$$
L(i)=exp(log L(i)) / \sum_{j=1}^{i}exp(log L(j))
$$

This approach has key properties:

 - Works even if streams are unrelated (no need for covariance/correlation).
 - Naturally supports missing sources (just omit them).
 - Supports “trust” scaling by weights $w_{s}$

**Bayesian posterior update**
$$
posterior_{t}​(i)∝L(i)⋅prior_{t}​(i)
$$

Where:
 - alpha = decay_factor (0..1)
 - small alpha → more reactive
 - larger alpha → more stable


## WORKING EXAMPLE

```python
class MultiSourceBayesianBeliefUpdater:
    """
    Generic discrete Bayesian belief updater with multiple evidence sources.

    You provide at each time step:
      evidence = {
        "macro":     np.array([... per hypothesis ...]),
        "sentiment": np.array([... per hypothesis ...]),
        "vol":       np.array([... per hypothesis ...]),
        ...
      }

    Each source vector is converted to a likelihood distribution over hypotheses.
    Likelihoods are combined multiplicatively (log-additively), then Bayes updates posterior.
    """

    def __init__(
        self,
        hypotheses,
        source_weights=None,
        decay_factor=0.05,
        weakening_threshold=0.2,
        temperature=1.0,
        eps=1e-12
    ):
        """
        Parameters:
        - hypotheses: list[str], discrete hypotheses H1..Hn
        - source_weights: dict[source -> weight] (>=0). Missing sources default to 1.0
        - decay_factor: (0..1) keeps inertia in priors
        - weakening_threshold: relative posterior drop threshold to flag weakening
        - temperature: softmax temperature for converting scores -> likelihood (higher = flatter)
        - eps: epsilon numerical stability constant. Very small positive number.
        """
        self.hypotheses = np.array(hypotheses, dtype=object)
        self.n = len(self.hypotheses)

        self.source_weights = dict(source_weights or {})
        self.decay_factor = float(decay_factor)
        self.weakening_threshold = float(weakening_threshold)
        self.temperature = float(temperature)
        self.eps = float(eps)

        self.prior = np.ones(self.n) / self.n
        self.posterior = self.prior.copy()

        self.history = []          # posterior snapshots
        self.source_history = []   # store per-source likelihoods (optional)

    @staticmethod
    def _softmax(x, temperature=1.0, eps=1e-12):
        x = np.asarray(x, dtype=float)
        x = x / max(temperature, eps)
        x = x - np.max(x)  # stability
        ex = np.exp(x)
        return ex / (np.sum(ex) + eps)

    def _scores_to_likelihood(self, scores):
        """
        Convert a vector of per-hypothesis scores into a likelihood distribution.

        Interpretation:
        - higher score => higher likelihood
        If your "evidence" is an error (lower is better), pass negative errors or transform before calling.
        """
        scores = np.asarray(scores, dtype=float)
        if scores.shape[0] != self.n:
            raise ValueError("Evidence vector length must match number of hypotheses.")
        return self._softmax(scores, temperature=self.temperature, eps=self.eps)

    def update(self, evidence_by_source):
        """
        evidence_by_source: dict[str, array-like of length n]
          Example:
            {
              "macro": [..],
              "sentiment": [..],
              "flows": [..]
            }
        """
        if not isinstance(evidence_by_source, dict) or len(evidence_by_source) == 0:
            raise ValueError("evidence_by_source must be a non-empty dict of source -> vector.")

        # Combine in log space: log P(D|H) = sum_s w_s * log P(D_s|H)
        log_like = np.zeros(self.n, dtype=float)
        per_source_likes = {}

        for source, vec in evidence_by_source.items():
            weight = float(self.source_weights.get(source, 1.0))
            if weight <= 0:
                continue  # ignore disabled sources cleanly

            like_s = self._scores_to_likelihood(vec)
            per_source_likes[source] = like_s

            log_like += weight * np.log(like_s + self.eps)

        # Convert back from log-likelihoods to combined likelihood
        log_like -= np.max(log_like)  # stability
        combined_likelihood = np.exp(log_like)
        combined_likelihood = combined_likelihood / (np.sum(combined_likelihood) + self.eps)

        # Bayes update
        numerator = combined_likelihood * self.prior
        self.posterior = numerator / (np.sum(numerator) + self.eps)

        # Inertia / forgetting
        self.prior = (1 - self.decay_factor) * self.posterior + self.decay_factor * self.prior

        self.history.append(self.posterior.copy())
        self.source_history.append(per_source_likes)

    def get_beliefs(self):
        df = pd.DataFrame({"hypothesis": self.hypotheses, "posterior": self.posterior})
        return df.sort_values("posterior", ascending=False).reset_index(drop=True)

    def detect_weakening(self, decimals=2):
        """
        Returns hypotheses whose posterior dropped sharply vs previous step.

        Output:
        [(hypothesis_label, drop_percentage)]
        """
        if len(self.history) < 2:
            return []

        prev_post = self.history[-2]
        curr_post = self.history[-1]

        drop = (prev_post - curr_post) / (prev_post + self.eps)

        mask = drop > self.weakening_threshold

        results = []
        for h, dp in zip(self.hypotheses[mask], drop[mask]):
            results.append((str(h), round(float(dp), decimals)))

        return results



hypotheses = ["Risk-On", "Neutral", "Risk-Off"]

weights = {
    "macro": 1.2,       # macro slightly more trusted
    "sentiment": 1.0,
    "volatility": 0.8   # vol helpful but not dominant
}


# Evidence stream: dict per timestep with multiple sources
stream = [
    {
        "macro":      [0.9, 0.7, 0.4],
        "sentiment":  [0.8, 0.6, 0.5],
        "volatility": [0.7, 0.7, 0.6],
    },
    {
        "macro":      [0.7, 0.8, 0.6],
        "sentiment":  [0.6, 0.7, 0.7],
        "volatility": [0.5, 0.7, 0.9],
    },
    {
        "macro":      [0.5, 0.7, 0.95],   # macro turning risk-off
        "sentiment":  [0.55, 0.65, 0.85],
        "volatility": [0.4, 0.6, 1.0],    # vol spike supports risk-off
    },
]

u = MultiSourceBayesianBeliefUpdater(
    hypotheses=hypotheses,
    source_weights=weights,
    decay_factor=0.08,
    weakening_threshold=0.25,
    temperature=1.0
)

for t, evidence in enumerate(stream, start=1):
    u.update(evidence)
    print(f"\nT{t} beliefs:")
    print(u.get_beliefs())

    wk = u.detect_weakening()
    if wk:
        print("Weakening detected:", wk)