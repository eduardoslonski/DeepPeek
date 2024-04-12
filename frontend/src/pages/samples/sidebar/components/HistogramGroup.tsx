import { fetchActivationsHistogram } from "@/api/activationsHistogramService"
import { Toggle } from "@/components/ui/toggle"
import Histogram from "@/components/custom/histogram"
import { useState } from "react"
import { useAtomValue } from "jotai"
import {
  ropeModeAtom,
  sampleWorkingSelectedAtom,
  selectedAttnHeadAtom,
  selectedLayerAtom,
  selectedTokensSamplesWorkingAtom,
} from "@/lib/atoms"
import { fetchHighlightActivation } from "@/api/highlightActivationService"
import { Input } from "@/components/ui/input"
import { keepPreviousData, useQuery } from "@tanstack/react-query"

interface HistogramGroupProps {
  type: string
}

type OutliersType = {
  name: string[]
  values: number[]
}

type HistogramData = {
  bins_histogram: number[]
  histogram: number[]
  median_histogram: number[]
  outliers: OutliersType
}

export default function HistogramGroup({ type }: HistogramGroupProps) {
  const [showOutliers, setShowOutliers] = useState<boolean>(true)
  const [highlightActivation, setHighlightActivation] = useState<number>()
  const selectedLayer = useAtomValue(selectedLayerAtom)
  const selectedAttnHead = useAtomValue(selectedAttnHeadAtom)
  const selectedTokensSamplesWorking = useAtomValue(
    selectedTokensSamplesWorkingAtom
  )
  const ropeMode = useAtomValue(ropeModeAtom)
  const sampleWorkingSelected = useAtomValue(sampleWorkingSelectedAtom)

  const { data, isLoading, isFetching } = useQuery({
    queryKey: [
      "fetchActivationsHistogram",
      sampleWorkingSelected,
      type,
      selectedLayer,
      selectedTokensSamplesWorking[sampleWorkingSelected],
      selectedAttnHead,
      ropeMode,
    ],
    queryFn: () =>
      fetchActivationsHistogram(
        sampleWorkingSelected,
        type,
        selectedLayer,
        selectedTokensSamplesWorking[sampleWorkingSelected],
        selectedAttnHead,
        ropeMode
      ),
    enabled: selectedTokensSamplesWorking[sampleWorkingSelected] !== undefined,
    placeholderData: keepPreviousData,
  })

  const { data: dataHighlightActivation } = useQuery({
    queryKey: [
      "fetchHighlightActivation",
      sampleWorkingSelected,
      type,
      selectedLayer,
      selectedAttnHead,
      selectedTokensSamplesWorking[sampleWorkingSelected],
      highlightActivation,
    ],
    queryFn: () =>
      fetchHighlightActivation({
        sampleIdx: sampleWorkingSelected,
        type: type,
        layer: selectedLayer,
        attnHead: selectedAttnHead,
        tokenIdx: selectedTokensSamplesWorking[sampleWorkingSelected],
        dimensionIdxTarget: highlightActivation,
      }),
    enabled:
      selectedTokensSamplesWorking[sampleWorkingSelected] !== undefined &&
      highlightActivation !== undefined,
    placeholderData: keepPreviousData,
  })

  const highlights =
    highlightActivation !== undefined && dataHighlightActivation !== undefined
      ? {
          name: [highlightActivation.toString()],
          values: [dataHighlightActivation[0]],
        }
      : { name: [], values: [] }

  const processData = (dataToProcess: HistogramData | null) => {
    const binsHistogram = dataToProcess?.bins_histogram ?? []
    const histogram = dataToProcess?.histogram ?? []
    const medianHistogram = dataToProcess?.median_histogram ?? []
    const outliers = showOutliers
      ? dataToProcess?.outliers ?? { name: [], values: [] }
      : { name: [], values: [] }

    const allValues = [
      ...outliers.values,
      ...highlights.values,
      ...binsHistogram,
    ]
    const xRange: [number, number] = [
      Math.min(...allValues) * 1.2,
      Math.max(...allValues) * 1.2,
    ]

    const yRange: [number, number] = [0, Math.max(...medianHistogram) * 1.5]

    return {
      xRange,
      yRange,
      binsHistogram,
      histogram,
      medianHistogram,
      outliers,
    }
  }

  const {
    xRange,
    yRange,
    binsHistogram,
    histogram,
    medianHistogram,
    outliers,
  } = processData(data)

  const handleKeyDownInput = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      const value = event.currentTarget.value
      if (value === "") {
        setHighlightActivation(undefined)
      } else {
        setHighlightActivation(Number(event.currentTarget.value))
      }
    }
  }

  return (
    <div>
      {selectedTokensSamplesWorking[sampleWorkingSelected] !== undefined &&
        !isLoading && (
          <div>
            <div className="flex justify-between items-center">
              <div className="flex inline-block gap-2">
                <span className="text-secondary-foreground text-xs font-semibold">
                  {type}
                </span>
                {(type === "q" ||
                  type === "k" ||
                  type === "q_rope" ||
                  type === "k_rope" ||
                  type === "v" ||
                  type === "o") && (
                  <span className="text-muted-foreground text-xs opacity-50">{`${
                    selectedAttnHead * 256
                  } - ${selectedAttnHead * 256 + 256 - 1}`}</span>
                )}
              </div>
              <div className="flex gap-2">
                <Input
                  className="w-24 h-7"
                  placeholder="highlights"
                  onKeyDown={handleKeyDownInput}
                  type="number"
                />
                <Toggle
                  pressed={showOutliers}
                  size={"xs"}
                  variant={"selecting"}
                  className="px-2"
                  onClick={() => setShowOutliers(!showOutliers)}
                >
                  Outliers
                </Toggle>
              </div>
            </div>
            <div className={`mt-4 ${isFetching && "opacity-40"}`}>
              <Histogram
                width={460}
                height={300}
                xRange={xRange}
                yRange={yRange}
                histograms={[
                  {
                    x: binsHistogram,
                    y: histogram,
                  },
                  {
                    x: binsHistogram,
                    y: medianHistogram,
                  },
                ]}
                outliers={outliers}
                highlights={highlights}
              />
            </div>
          </div>
        )}
    </div>
  )
}
