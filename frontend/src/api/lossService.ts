export const fetchLoss = async (sampleIdx: number) => {
  const response = await fetch(`/api/samples/loss?sample_idx=${sampleIdx}`)
  const data = await response.json()
  return data
}
