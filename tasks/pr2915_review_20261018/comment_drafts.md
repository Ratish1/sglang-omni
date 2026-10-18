# Comment drafts — not staged or posted

## stages.py:1243

Use the PyTorch fallback for thinker min-p batches too. FlashInfer applies top-p after renormalizing top-k; the old path uses the original probabilities. This changes allowed tokens even for min_p=0 rows when another row enables min-p. Add a mixed-batch case that checks the allowed tokens.

## Optional description comment

Show full-corpus WER and the count of samples above 50% WER beside the filtered WER for both arms.
