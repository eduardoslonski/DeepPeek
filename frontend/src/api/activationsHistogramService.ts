export const fetchActivationsHistogram = async (
  sampleIdx: number,
  type: string,
  layer: number,
  tokenIdx: number | undefined,
  attnHead: number,
  ropeMode: string
) => {
  const response = await fetch(
    `/api/samples/activations/histogram/${type}/${layer}/${tokenIdx}?attn_head=${attnHead}&rope_mode=${ropeMode}&sample_idx=${sampleIdx}`
  )
  const data = await response.json()
  return data
}
