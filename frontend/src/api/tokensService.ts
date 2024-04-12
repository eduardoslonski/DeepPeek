export const fetchTokens = async (sampleIdx: number, sample: string) => {
  const response = await fetch(
    `/api/samples/tokens?sample=${encodeURIComponent(
      sample
    )}&sample_idx=${sampleIdx}`
  )
  const data = await response.json()
  return data
}
