export const fetchAttentionOpposite = async (
  sampleIdx: number,
  layer: number,
  attn_head: number,
  token: number | undefined
) => {
  const response = await fetch(
    `/api/samples/attention_opposite/${layer}/${attn_head}/${token}?sample_idx=${sampleIdx}`
  )
  const data = await response.json()
  return data
}
