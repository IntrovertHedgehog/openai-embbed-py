## Embbeding
- Quick, accurate answer
- Cheap
- Cannot extend beyond provided questions/answer
- Need more data to be useful (to answer all the question might be asked by the users)

## Index Store (Llama Index)
- Slower (~18s/prompt before optimisation)
- Consumes more resource (tokens), therefore much more expensive
- Can extend beyond provided context: response synthesis
- Less accurate than embbeding, but acceptable for most cases

## Proposed
- Enhance Embbeding with response engineering
- Combine LlamaIndex with Embbeding to cover wide range of cases
- Optimising LlamaIndex to be cheaper and faster

### Index Optimization
- Vector Index seems most appropriate for semantic search task such as this
- [HyDE](https://gpt-index.readthedocs.io/en/latest/how_to/query/query_transformations.html#hyde-hypothetical-document-embeddings) is tried, but it does not improve the current model too much, plus it consumes more time and tokens
- Single-step decomposition is potentially useful. It can answer more complex queries by transform the query into subquestion and check against the index. The current model does not have this capability, but will be integrated in the future to cover more use cases.
- 