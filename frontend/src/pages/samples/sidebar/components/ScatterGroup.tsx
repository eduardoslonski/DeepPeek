import { fetchActivationsValues } from "@/api/activationsValueService"
import ScatterPlot from "@/components/custom/scatterPlot"
import { Input } from "@/components/ui/input"
import {
  hasForwardDataAtom,
  sampleWorkingSelectedAtom,
  selectedAttnHeadAtom,
  selectedLayerAtom,
  selectedTokensSamplesWorkingAtom,
} from "@/lib/atoms"
import { keepPreviousData, useQuery } from "@tanstack/react-query"
import { useAtomValue } from "jotai"
import { useState } from "react"

interface ScatterGroupProps {
  typeActivation: string
}
export default function ScatterGroup({ typeActivation }: ScatterGroupProps) {
  const selectedLayer = useAtomValue(selectedLayerAtom)
  const selectedAttnHead = useAtomValue(selectedAttnHeadAtom)
  const selectedTokensSamplesWorking = useAtomValue(
    selectedTokensSamplesWorkingAtom
  )
  const sampleWorkingSelected = useAtomValue(sampleWorkingSelectedAtom)

  const selectedToken = selectedTokensSamplesWorking[sampleWorkingSelected]
  const [
    activationsActivationIdxSelected,
    setActivationsActivationIdxSelected,
  ] = useState<number>()

  const hasForwardData = useAtomValue(hasForwardDataAtom)

  const { data: dataActivationsValues, isFetching: isFetchingValues } =
    useQuery({
      queryKey: [
        "dataActivationsValues",
        sampleWorkingSelected,
        typeActivation,
        selectedLayer,
        selectedAttnHead,
        activationsActivationIdxSelected,
      ],
      queryFn: () =>
        fetchActivationsValues(
          sampleWorkingSelected,
          typeActivation,
          selectedLayer,
          selectedAttnHead,
          activationsActivationIdxSelected as number
        ),
      enabled: activationsActivationIdxSelected !== undefined,
      placeholderData: keepPreviousData,
    })

  const values = dataActivationsValues
    ? dataActivationsValues.activations
    : undefined
  const xRange: [number, number] = dataActivationsValues && [
    values.length * -0.02,
    values.length + values.length * 0.02,
  ]
  const yRange: [number, number] = dataActivationsValues && [
    Math.min(...values) * 1.2,
    Math.max(...values) * 1.2,
  ]
  const selectedTokensToHighlight: [number, number][] =
    selectedToken !== undefined && dataActivationsValues
      ? [[selectedToken, dataActivationsValues.activations[selectedToken]]]
      : []

  const markerColors: string[] = values && [
    ...values.map(() => (selectedToken !== undefined ? "#87c7f5" : "#1f77b4")),
    ...selectedTokensToHighlight.map(() => "red"),
  ]

  const handleKeyDownInputActivation = (
    event: React.KeyboardEvent<HTMLInputElement>
  ) => {
    if (event.key === "Enter") {
      const value = event.currentTarget.value
      if (value === "") {
        setActivationsActivationIdxSelected(undefined)
      } else {
        setActivationsActivationIdxSelected(Number(event.currentTarget.value))
      }
    }
  }

  return (
    <>
      {hasForwardData[sampleWorkingSelected] && (
        <div>
          <div className="flex justify-between">
            <span className="text-secondary-foreground text-xs font-semibold">
              {typeActivation}
            </span>
            <Input
              className="w-24 h-7"
              placeholder="activation"
              onKeyDown={handleKeyDownInputActivation}
              type="number"
            />
          </div>
          {dataActivationsValues && (
            <div className={`mt-4 ${isFetchingValues && "opacity-40"}`}>
              <ScatterPlot
                values={values}
                markerColors={markerColors}
                xRange={xRange}
                yRange={yRange}
                selectedTokensToHighlight={selectedTokensToHighlight}
                height={350}
                width={460}
              />
            </div>
          )}
        </div>
      )}
    </>
  )
}
