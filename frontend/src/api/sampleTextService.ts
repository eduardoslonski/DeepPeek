export const fetchSampleText = async (dataset: string, id: number) => {
  const response = await fetch(`/api/general/sample-text/${dataset}/${id}`)
  const data = await response.json()
  return data
}
