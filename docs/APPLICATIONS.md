# Application scope

The service targets **sparse access to trainable embedding tables**: a training
batch selects a small set of rows from a much larger parameter table. The selected
row vectors are usually dense. This access pattern appears in several task families.

| Task family | Typical lookup keys | Training work served |
|---|---|---|
| Recommendation and advertising | User, item, application, campaign and categorical-feature IDs | Retrieve embeddings and update the rows touched by an interaction batch |
| Knowledge graph embedding | Entity and relation IDs | Train entity/relation vectors for link prediction and relation scoring |
| Graph learning with trainable node embeddings | Node or edge IDs | Read and update the embeddings involved in a sampled mini-batch |
| NLP embedding learning | Word or token IDs | Train lookup tables in word-embedding and suitable sparse-update models |
| High-cardinality tabular learning | Account, device, merchant or other categorical IDs | Learn categorical representations for classification, sequence modeling and anomaly-related applications |

The first integration retains the existing Entropy training framework. The
service boundary uses tables, keys, dimensions, pooling/sequence metadata and
gradient batches, so it does not depend on video-specific identifiers.

## Related sparse representations

Learned sparse retrieval, such as SPLADE, produces high-dimensional query and
document vectors with many zero entries for lexical retrieval and inverted
indexes. This is another important use of sparsity. Its retrieval representation
is different from the row-sparse parameter access that motivates this service.

## Sources

1. Cheng, H.-T., et al. *Wide & Deep Learning for Recommender Systems.* 2016.
   [Google Research](https://research.google/pubs/wide-deep-learning-for-recommender-systems/).
2. McMahan, H. B., et al. *Ad Click Prediction: a View from the Trenches.* KDD, 2013.
   [Google Research](https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/).
3. Zheng, D., et al. *DGL-KE: Training Knowledge Graph Embeddings at Scale.* SIGIR, 2020.
   [Amazon Science](https://www.amazon.science/publications/dgl-ke-training-knowledge-graph-embeddings-at-scale),
   [distributed parameter-server training](https://github.com/awslabs/dgl-ke/blob/master/docs/source/dist_train.rst).
4. DGL. *Distributed Node Embedding.*
   [Documentation](https://www.dgl.ai/dgl_docs/api/python/dgl.distributed.html).
5. PyTorch. *Embedding.*
   [Documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Embedding.html).
6. Li, A., Sankararaman, A., and Narayanaswamy, M. *Probabilistic Hash Embeddings for Online Learning of Categorical Features.* 2026.
   [Amazon Science](https://www.amazon.science/publications/probabilistic-hash-embeddings-for-online-learning-of-categorical-features).
7. Formal, T., Piwowarski, B., and Clinchant, S. *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.* SIGIR, 2021.
   [Paper](https://arxiv.org/abs/2107.05720).
