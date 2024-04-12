export const fetchSamplesToSelect = async () => {
  const response = await fetch(`/api/general/samples-to-select`)
  const data = await response.json()
  return data
}
