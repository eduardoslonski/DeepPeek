export const fetchSimilaritiesPrevious = async (
  sampleIdx: number,
  type: string,
  layer: number,
  attnHead: number,
  ropeMode: string
) => {
  const response = await fetch(
    `/api/samples/similarities_previous/${type}/${layer}?attn_head=${attnHead}&rope_mode=${ropeMode}&sample_idx=${sampleIdx}`
  )
  const data = await response.json()
  return data
}
