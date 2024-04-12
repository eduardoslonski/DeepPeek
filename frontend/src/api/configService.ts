export const fetchConfig = async () => {
  const response = await fetch(`/api/general/model-config`)
  const data = await response.json()
  return data
}
