import {
  sampleWorkingSelectedAtom,
  selectedTokensSamplesWorkingAtom,
} from "@/lib/atoms"
import { useAtomValue, useSetAtom } from "jotai"
import Plot, { PlotParams } from "react-plotly.js"

interface ScatterProps {
  height?: number
  width?: number
  values: number[]
  markerColors: string[]
  xRange: [number, number]
  yRange: [number, number]
  selectedTokensToHighlight: [number, number][]
}

export default function ScatterPlot({
  height = 400,
  width = 460,
  values,
  markerColors,
  xRange,
  yRange,
  selectedTokensToHighlight,
}: ScatterProps) {
  const setSelectedTokensSamplesWorking = useSetAtom(
    selectedTokensSamplesWorkingAtom
  )
  const sampleWorkingSelected = useAtomValue(sampleWorkingSelectedAtom)

  const layout: Partial<PlotParams["layout"]> = {
    width: width,
    height: height,
    showlegend: false,
    margin: { t: 0, l: 40, r: 20, b: 20 },
    xaxis: {
      range: xRange,
      showgrid: false,
      zeroline: false,
    },
    yaxis: { range: yRange, showgrid: false, zeroline: false },
    shapes: [
      {
        type: "line",
        x0: xRange[0],
        y0: 0,
        x1: xRange[1],
        y1: 0,
        line: {
          color: "rgba(240, 240, 240, 0.7)",
          width: 2,
        },
      },
    ],
  }

  const config: Partial<PlotParams["config"]> = {
    displayModeBar: false,
  }

  const xHighlights = selectedTokensToHighlight.map((pair) => pair[0])
  const yHighlights = selectedTokensToHighlight.map((pair) => pair[1])

  const data: PlotParams["data"] = [
    {
      x: [
        ...Array.from({ length: values.length }, (_, index) => index),
        ...xHighlights,
      ],
      y: [...values, ...yHighlights],
      type: "scatter",
      mode: "markers",
      marker: { color: markerColors },
    },
  ]

  const handlePlotClick = (eventData: any) => {
    if (eventData.points && eventData.points.length > 0) {
      const xCoordinate = eventData.points[0].x
      setSelectedTokensSamplesWorking((prevList) =>
        prevList.map((item, idx) =>
          idx === sampleWorkingSelected
            ? item === xCoordinate
              ? undefined
              : xCoordinate
            : item
        )
      )
    }
  }

  return (
    <Plot
      data={data}
      layout={layout}
      config={config}
      onClick={handlePlotClick}
    />
  )
}
