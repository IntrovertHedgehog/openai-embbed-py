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
