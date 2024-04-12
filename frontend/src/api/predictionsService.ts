export const fetchPredictions = async (sampleIdx: number, top: number) => {
  const response = await fetch(
    `/api/samples/predictions?top=${top}&sample_idx=${sampleIdx}`
  )
  const data = await response.json()
  return data
}
