# Decision

Ship Approach A first. Later models replace it only if they raise hold-out macro F0.5.

- Task is bipartite linkage twice: Source 1 to Source 2, and Source 1 to Source 3.
- Score is macro F0.5. Empty predictions on singletons score 1.0, so the threshold is tuned for F0.5, not top-k retrieval.
- Country is a soft feature. It is never a hard filter and is not one-hot to the training labels.
- No live lookups: no geocoders, registries, or entity APIs.
- `candidate_pairs.tsv` is the set the matcher scores, and every submitted match must be in that set.
- Model license target if a neural model is added later: MIT or Apache-2.0, at most 8B parameters. Llama and Qwen3-Embedding are out on license.

## Approach A (in progress)

1. Normalize names and addresses with the abbreviation patterns in the problem statement.
2. Block with a token inverted index (name tokens, a short address-token tail, and a 5-character name prefix). Document frequency is capped so huge tokens do not explode the candidate set. Same country adds a bonus only.
3. Score each surviving pair with Jaro-Winkler, ratio, token Jaccard, length delta, country agreement, and source flag.
4. Fit LightGBM on blocking candidates. Positives are ground-truth IDs inside that candidate set.
5. Choose the probability cutoff on a held-out slice of Source 1 entities.

Entry point: `python -m src.pipeline --max-entities 20000` from `code/business_entity_resolution` with `PYTHONPATH` set. Full training over every Source 1 entity is the same command with a higher cap once a short run looks sane.
