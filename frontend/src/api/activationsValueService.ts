export const fetchActivationsValues = async (
  sampleIdx: number,
  type: string,
  layer: number,
  attnHead: number,
  activationIdx: number
) => {
  const response = await fetch(
    `/api/samples/activations_values/${type}/${layer}?attn_head=${attnHead}&activation_idx=${activationIdx}&sample_idx=${sampleIdx}`
  )
  const data = await response.json()
  return data
}
