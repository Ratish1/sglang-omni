# Comment drafts — not staged or posted

## stages.py:1243

Preserve the existing joint filtering for min-p batches. SGLang's FlashInfer min-p path renormalizes top-k before applying top-p, while the PyTorch path uses the original probabilities. This changes which tokens are allowed, including for rows with min_p=0 when another request in the batch enables min-p. Route these batches through the existing PyTorch fallback, or preserve the joint support on the GPU, and add a case for the mixed batch.

## Optional description comment

Include full-corpus WER and the number of samples above 50% WER for both arms alongside the filtered WER.
