export const calculateColorAttention = (
  value: number,
  opacity: number,
  isAltPressed: boolean,
  altMultiplyFactorAttention: number
) => {
  if (isAltPressed) {
    value = value * altMultiplyFactorAttention
  }
  let r, g, b
  r = 255
  g = b = Math.floor(255 * (1 - value))

  return `rgba(${r}, ${g}, ${b}, ${opacity})`
}

export const calculateColorLoss = (value: number, opacity: number) => {
  let r, g, b
  r = 255
  g = b = Math.floor(255 * (1 - Math.min(1, value / 10)))

  return `rgba(${r}, ${g}, ${b}, ${opacity})`
}

const interpolateColor = (
  startColor: number[],
  endColor: number[],
  value: number
) => {
  return startColor
    .map((color, index) => {
      const interpolated = color + (endColor[index] - color) * value
      return Math.round(interpolated)
    })
    .join(", ")
}

export const calculateColorSimilarities = (
  value: number,
  opacity: number,
  normalizationMode: string
) => {
  const red = [255, 0, 0]
  const white = [255, 255, 255]
  const blue = [0, 0, 255]

  let color
  if (value < (normalizationMode === "default" ? 0 : 0.5)) {
    color = interpolateColor(
      red,
      white,
      normalizationMode === "default" ? value + 1 : (0.5 - value) * 2
    )
  } else {
    color = interpolateColor(
      white,
      blue,
      normalizationMode === "default" ? value : (value - 0.5) * 2
    )
  }

  return `rgb(${color}, ${opacity})`
}

export const calculateColorActivationsSum = (
  value: number,
  opacity: number,
  normalizationMode: string
) => {
  let r, g, b
  b = 255
  g = r = Math.floor(
    255 *
      (1 -
        (normalizationMode === "default" ? Math.min(1, value / 1000) : value))
  )

  return `rgba(${r}, ${g}, ${b}, ${opacity})`
}

export const calculateColorActivationsValues = (
  value: number,
  opacity: number,
  limits: [number, number],
  divider: number,
  normalizationMode: string
) => {
  const red = [255, 0, 0]
  const white = [255, 255, 255]
  const blue = [0, 0, 255]

  let color
  if (value < divider) {
    color = interpolateColor(
      white,
      red,
      1 - (value - limits[0]) / (divider - limits[0])
    )
  } else {
    color = interpolateColor(
      white,
      blue,
      (value - divider) / (limits[1] - divider)
    )
  }

  return `rgb(${color}, ${opacity})`
}
