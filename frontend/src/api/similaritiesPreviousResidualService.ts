export const fetchSimilaritiesPreviousResidual = async (
  sampleIdx: number,
  type: string,
  layer: number
) => {
  const response = await fetch(
    `/api/samples/similarities_previous_residual/${type}/${layer}?sample_idx=${sampleIdx}`
  )
  const data = await response.json()
  return data
}
