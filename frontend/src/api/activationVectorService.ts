export const fetchActivationVector = async (
  sampleIdx: number,
  type: string,
  layer: number,
  attnHead: number,
  tokenIdx: number
) => {
  const response = await fetch(
    `/api/samples/activation_vector/${type}/${layer}?attn_head=${attnHead}&token_idx=${tokenIdx}&sample_idx=${sampleIdx}`
  )
  const data = await response.json()
  return data
}
