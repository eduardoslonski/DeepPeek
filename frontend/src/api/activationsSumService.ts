export const fetchActivationsSum = async (
  sampleIdx: number,
  type: string,
  layer: number,
  attnHead: number
) => {
  const response = await fetch(
    `/api/samples/activations_sum/${type}/${layer}?attn_head=${attnHead}&sample_idx=${sampleIdx}`
  )
  const data = await response.json()
  return data
}
