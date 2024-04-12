export const fetchHighlightActivation = async ({
  sampleIdx,
  type,
  layer,
  attnHead,
  tokenIdx,
  dimensionIdxTarget,
}: {
  sampleIdx: number
  type: string
  layer?: number
  attnHead?: number
  tokenIdx?: number
  dimensionIdxTarget?: number
}) => {
  const response = await fetch(
    `/api/samples/highlight/${type}/${layer}/${tokenIdx}/${dimensionIdxTarget}?attn_head=${attnHead}&sample_idx=${sampleIdx}`
  )
  const data = await response.json()
  return data
}
