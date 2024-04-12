export const fetchSimilaritiesTokens = async (
  sampleIdx: number,
  type: string,
  layer: number,
  token: number | undefined,
  attnHead: number,
  ropeMode: string
) => {
  const response = await fetch(
    `/api/samples/similarities_tokens/${type}/${layer}/${token}?attn_head=${attnHead}&rope_mode=${ropeMode}&sample_idx=${sampleIdx}`
  )
  const data = await response.json()
  return data
}
