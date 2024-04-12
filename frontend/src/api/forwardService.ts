export const fetchForward = async (sampleIdx: number) => {
  const response = await fetch(`/api/samples/forward?sample_idx=${sampleIdx}`)
  const data = await response.json()
  return data
}
