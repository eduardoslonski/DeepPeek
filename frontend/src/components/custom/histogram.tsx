import Plot, { PlotParams } from "react-plotly.js"

type Range = [number, number]

interface HistogramType {
  x: number[]
  y: number[]
}

type OutliersType = {
  name: string[]
  values: number[]
}

type HighlightsType = {
  name: string[]
  values: number[]
}

interface HistogramProps {
  width?: number
  height?: number
  histograms: HistogramType[]
  opacity?: number
  xRange: Range
  yRange: Range
  outliers: OutliersType
  highlights: HighlightsType
}

export default function Histogram({
  width = 600,
  height = 350,
  histograms,
  opacity = 0.6,
  xRange,
  yRange,
  outliers,
  highlights,
}: HistogramProps) {
  const outlierShapes = outliers.values.map((outlier) => ({
    type: "line" as const,
    x0: outlier,
    y0: 0,
    x1: outlier,
    y1: yRange[1],
    line: {
      color: "#ffa8a8",
      width: 2,
    },
  }))

  const highlightsShapes = highlights.values.map((highlight) => ({
    type: "line" as const,
    x0: highlight,
    y0: 0,
    x1: highlight,
    y1: yRange[1],
    line: {
      color: "#a8bbff",
      width: 2,
    },
  }))

  const createOutlierHoverPoints = (
    outlierValue: number,
    maxHeight: number
  ) => {
    const yPoints = Array.from({ length: 10 }, (_, i) => (i * maxHeight) / 10)
    return {
      x: Array(yPoints.length).fill(outlierValue),
      y: yPoints,
      mode: "markers" as const,
      type: "scatter" as const,
      marker: {
        size: 1,
        opacity: 0,
      },
      hoverinfo: "text" as const,
      text: Array(yPoints.length).fill(
        `${
          outliers.name[outliers.values.indexOf(outlierValue)]
        } (${outlierValue.toFixed(2)})`
      ),
      showlegend: false,
      hoverlabel: {
        bgcolor: "#ffa8a8",
        ffont: {
          size: 12,
          family: "Arial, sans-serif",
          color: "black",
          style: "bold",
        },
      },
    }
  }

  const createHighlightHoverPoints = (
    highlightValue: number,
    maxHeight: number
  ) => {
    const yPoints = Array.from({ length: 10 }, (_, i) => (i * maxHeight) / 10)
    return {
      x: Array(yPoints.length).fill(highlightValue),
      y: yPoints,
      mode: "markers" as const,
      type: "scatter" as const,
      marker: {
        size: 1,
        opacity: 0,
      },
      hoverinfo: "text" as const,
      text: Array(yPoints.length).fill(
        `${
          highlights.name[highlights.values.indexOf(highlightValue)]
        } (${highlightValue.toFixed(2)})`
      ),
      showlegend: false,
      hoverlabel: {
        bgcolor: "#a8bbff",
        ffont: {
          size: 12,
          family: "Arial, sans-serif",
          color: "black",
          style: "bold",
        },
      },
    }
  }

  const outlierHoverTraces = outliers.values.map((outlierValue) =>
    createOutlierHoverPoints(outlierValue, yRange[1])
  )
  const highlightsHoverTraces = highlights.values.map((highlightValue) =>
    createHighlightHoverPoints(highlightValue, yRange[1])
  )

  const layout: Partial<PlotParams["layout"]> = {
    bargap: 0.05,
    barmode: "overlay",
    xaxis: { range: xRange, showgrid: false },
    yaxis: { range: yRange, showgrid: false },
    shapes: [...outlierShapes, ...highlightsShapes],
    width: width,
    height: height,
    showlegend: false,
    margin: { t: 0, l: 40, r: 20, b: 20 },
  }

  const config: Partial<PlotParams["config"]> = {
    displayModeBar: false,
  }

  const data: PlotParams["data"] = [
    ...histograms.map((hist) => ({
      type: "bar" as const,
      opacity: opacity,
      x: hist.x,
      y: hist.y,
    })),
    ...outlierHoverTraces,
    ...highlightsHoverTraces,
  ]

  return <Plot data={data} layout={layout} config={config} />
}
