Project Description:
This project addresses the DSTC12 Track 2 challenge: Controllable Conversational Theme Detection.
The task involves detecting high-level themes in user utterances by:

1. Training a semantic embedding model on similarity constraints.
2. Clustering semantically similar utterances.
3. Using a large language model (LLM) to generate a human-readable label for each cluster.

Problem Tackled:
Given a set of utterances with human-labeled themes and preference pairs (should-link / cannot-link), the objective is to build a model that:
- Embeds utterances so similar ones are close in vector space.
- Clusters utterances without supervision.
- Labels each cluster with a short, action-oriented theme.
This setup simulates zero-shot theme detection on a new domain by training on Finance and evaluating on Travel.
