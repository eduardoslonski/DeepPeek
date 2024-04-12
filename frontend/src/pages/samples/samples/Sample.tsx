import { useQuery, useQueryClient } from "@tanstack/react-query"
import Menu from "./Menu"
import { fetchForward } from "@/api/forwardService"
import { useAtom, useAtomValue, useSetAtom } from "jotai"
import {
  activationSelectedAtom,
  activationsActivationIdxSelectedAtom,
  activationsTypeSelectedAtom,
  attentionOppositeAtom,
  hasForwardDataAtom,
  isAltPressedAtom,
  isCommandPressedAtom,
  isCtrlPressedAtom,
  modeVisualizationAtom,
  normalizationModeAtom,
  ropeModeAtom,
  sampleWorkingSelectedAtom,
  samplesWorkingTextAtom,
  samplesWorkingTokensAtom,
  selectedAttnHeadAtom,
  selectedLayerAtom,
  selectedTokensSamplesWorkingAtom,
  showPredictionsAtom,
  similaritiesTypeSelectedAtom,
} from "@/lib/atoms"
import { useEffect, useState } from "react"
import { fetchTokens } from "@/api/tokensService"
import TokenElement from "./components/TokenElement"
import Predictions from "./components/Predictions"
import { fetchAttention } from "@/api/attentionService"
import { fetchLoss } from "@/api/lossService"
import { fetchSimilaritiesPrevious } from "@/api/similaritiesPreviousService"
import { fetchSimilaritiesTokens } from "@/api/similaritiesTokensService"
import { fetchActivationsValues } from "@/api/activationsValueService"
import { fetchActivationsSum } from "@/api/activationsSumService"
import { fetchSimilaritiesPreviousResidual } from "@/api/similaritiesPreviousResidualService"
import { fetchAttentionOpposite } from "@/api/attentionOppositeService"

export default function Sample() {
  const queryClient = useQueryClient()
  const sampleWorkingSelected = useAtomValue(sampleWorkingSelectedAtom)
  const [hasForwardData, setHasForwardData] = useAtom(hasForwardDataAtom)
  const samplesWorkingText = useAtomValue(samplesWorkingTextAtom)

  const [samplesWorkingTokens, setSamplesWorkingTokens] = useAtom(
    samplesWorkingTokensAtom
  )
  const setIsCtrlPressed = useSetAtom(isCtrlPressedAtom)
  const setIsAltPressed = useSetAtom(isAltPressedAtom)
  const setIsCommandPressed = useSetAtom(isCommandPressedAtom)
  const [selectedTokensSamplesWorking, setSelectedTokensSamplesWorking] =
    useAtom(selectedTokensSamplesWorkingAtom)
  const [startedFetchingForward, setStartedFetchingForward] =
    useState<boolean>(false)
  const [startedFetchingTokens, setStartedFetchingTokens] =
    useState<boolean>(false)

  const showPredictions = useAtomValue(showPredictionsAtom)
  const modeVisualization = useAtomValue(modeVisualizationAtom)
  const ropeMode = useAtomValue(ropeModeAtom)
  const similaritiesTypeSelected = useAtomValue(similaritiesTypeSelectedAtom)
  const normalizationMode = useAtomValue(normalizationModeAtom)
  const activationsActivationIdxSelected = useAtomValue(
    activationsActivationIdxSelectedAtom
  )
  const attentionOpposite = useAtomValue(attentionOppositeAtom)
  const activationSelected = useAtomValue(activationSelectedAtom)
  const activationsTypeSelected = useAtomValue(activationsTypeSelectedAtom)

  const selectedLayer = useAtomValue(selectedLayerAtom)
  const selectedAttnHead = useAtomValue(selectedAttnHeadAtom)

  const selectedToken = selectedTokensSamplesWorking[sampleWorkingSelected]

  const {
    data: dataTokens,
    refetch: refetchTokens,
    isFetching: isFetchingTokens,
  } = useQuery({
    queryKey: [
      "dataTokens",
      sampleWorkingSelected,
      samplesWorkingText[sampleWorkingSelected],
    ],
    queryFn: () =>
      fetchTokens(
        sampleWorkingSelected,
        samplesWorkingText[sampleWorkingSelected]
      ),
    enabled: false,
  })

  useEffect(() => {
    if (isFetchingTokens) {
      setStartedFetchingTokens(true)
    } else {
      if (startedFetchingTokens) {
        setSamplesWorkingTokens((currentList) => {
          const updatedList = [...currentList]
          if (sampleWorkingSelected < updatedList.length) {
            updatedList[sampleWorkingSelected] = dataTokens.tokens
          } else {
            updatedList[sampleWorkingSelected] = dataTokens.tokens
          }
          return updatedList
        })
        setHasForwardData((currentList) => {
          const updatedList = [...currentList]
          if (sampleWorkingSelected < updatedList.length) {
            updatedList[sampleWorkingSelected] = false
          } else {
            updatedList[sampleWorkingSelected] = false
          }
          return updatedList
        })
        if (hasForwardData[sampleWorkingSelected] === false) {
          refetchForward()
        }
        setStartedFetchingTokens(false)
        if (selectedTokensSamplesWorking.length <= sampleWorkingSelected) {
          setSelectedTokensSamplesWorking((prevList) => [
            ...prevList,
            undefined,
          ])
        } else {
          setSelectedTokensSamplesWorking((prevList) =>
            prevList.map((item, idx) =>
              idx === sampleWorkingSelected ? undefined : item
            )
          )
        }
      }
    }
  }, [isFetchingTokens])

  useEffect(() => {
    if (samplesWorkingText[sampleWorkingSelected]) {
      queryClient.removeQueries({
        queryKey: ["dataTokens", sampleWorkingSelected],
      })
      refetchTokens()
    }
  }, [samplesWorkingText])

  const { isFetching: isFetchingForward, refetch: refetchForward } = useQuery({
    queryKey: ["dataForward", sampleWorkingSelected],
    queryFn: () => fetchForward(sampleWorkingSelected),
    enabled: false,
  })

  useEffect(() => {
    if (isFetchingForward) {
      setStartedFetchingForward(true)
    } else {
      if (startedFetchingForward) {
        setHasForwardData((currentList) => {
          const updatedList = [...currentList]
          updatedList[sampleWorkingSelected] = true
          return updatedList
        })
        queryClient.removeQueries({
          queryKey: ["dataAttention", sampleWorkingSelected],
        })
        queryClient.removeQueries({
          queryKey: ["dataAttentionOpposite", sampleWorkingSelected],
        })
        queryClient.removeQueries({
          queryKey: ["dataLoss", sampleWorkingSelected],
        })
        queryClient.removeQueries({
          queryKey: ["dataSimilaritiesTokens", sampleWorkingSelected],
        })
        queryClient.removeQueries({
          queryKey: ["dataSimilaritiesPrevious", sampleWorkingSelected],
        })
        queryClient.removeQueries({
          queryKey: ["dataSimilaritiesPreviousResidual", sampleWorkingSelected],
        })
        queryClient.removeQueries({
          queryKey: ["dataActivationsValues", sampleWorkingSelected],
        })
        queryClient.removeQueries({
          queryKey: ["dataActivationsSum", sampleWorkingSelected],
        })
        queryClient.removeQueries({
          queryKey: ["dataActivationsSum", sampleWorkingSelected],
        })
        queryClient.removeQueries({
          queryKey: ["dataActivationsSum", sampleWorkingSelected],
        })
        queryClient.removeQueries({
          queryKey: ["dataActivationsSum", sampleWorkingSelected],
        })
        setStartedFetchingForward(false)
      }
    }
  }, [isFetchingForward])

  const { data: dataAttention } = useQuery({
    queryKey: [
      "dataAttention",
      sampleWorkingSelected,
      selectedLayer,
      selectedAttnHead,
      selectedTokensSamplesWorking[sampleWorkingSelected],
    ],
    queryFn: () =>
      fetchAttention(
        sampleWorkingSelected,
        selectedLayer,
        selectedAttnHead,
        selectedTokensSamplesWorking[sampleWorkingSelected]
      ),
    enabled:
      modeVisualization === "attention" &&
      selectedTokensSamplesWorking[sampleWorkingSelected] !== undefined &&
      attentionOpposite === false,
  })

  const { data: dataAttentionOpposite } = useQuery({
    queryKey: [
      "dataAttentionOpposite",
      sampleWorkingSelected,
      selectedLayer,
      selectedAttnHead,
      selectedTokensSamplesWorking[sampleWorkingSelected],
    ],
    queryFn: () =>
      fetchAttentionOpposite(
        sampleWorkingSelected,
        selectedLayer,
        selectedAttnHead,
        selectedTokensSamplesWorking[sampleWorkingSelected]
      ),
    enabled:
      modeVisualization === "attention" &&
      selectedTokensSamplesWorking[sampleWorkingSelected] !== undefined &&
      attentionOpposite === true,
  })

  const { data: dataLoss } = useQuery({
    queryKey: ["dataLoss", sampleWorkingSelected],
    queryFn: () => fetchLoss(sampleWorkingSelected),
    enabled: modeVisualization === "loss",
  })

  const { data: dataSimilaritiesTokens } = useQuery({
    queryKey: [
      "dataSimilaritiesTokens",
      sampleWorkingSelected,
      activationSelected,
      selectedLayer,
      selectedToken,
      selectedAttnHead,
      ropeMode,
    ],
    queryFn: () =>
      fetchSimilaritiesTokens(
        sampleWorkingSelected,
        activationSelected,
        selectedLayer,
        selectedToken,
        selectedAttnHead,
        ropeMode
      ),
    enabled:
      modeVisualization === "similarities" &&
      similaritiesTypeSelected === "tokens" &&
      activationSelected !== "" &&
      selectedToken !== undefined,
  })

  const { data: dataSimilaritiesPrevious } = useQuery({
    queryKey: [
      "dataSimilaritiesPrevious",
      sampleWorkingSelected,
      activationSelected,
      selectedLayer,
      selectedAttnHead,
      ropeMode,
    ],
    queryFn: () =>
      fetchSimilaritiesPrevious(
        sampleWorkingSelected,
        activationSelected,
        selectedLayer,
        selectedAttnHead,
        ropeMode
      ),
    enabled:
      modeVisualization === "similarities" &&
      activationSelected !== "" &&
      selectedLayer > 0 &&
      similaritiesTypeSelected === "previous",
  })

  const { data: dataSimilaritiesPreviousResidual } = useQuery({
    queryKey: [
      "dataSimilaritiesPreviousResidual",
      sampleWorkingSelected,
      activationSelected,
      selectedLayer,
    ],
    queryFn: () =>
      fetchSimilaritiesPreviousResidual(
        sampleWorkingSelected,
        activationSelected,
        selectedLayer
      ),
    enabled:
      modeVisualization === "similarities" &&
      (activationSelected === "dense_attention" ||
        activationSelected === "dense_attention_residual" ||
        activationSelected === "mlp_4_to_h" ||
        activationSelected === "output") &&
      selectedLayer > 0 &&
      similaritiesTypeSelected === "previousResidual",
  })

  const { data: dataActivationsValues } = useQuery({
    queryKey: [
      "dataActivationsValues",
      sampleWorkingSelected,
      activationSelected,
      selectedLayer,
      selectedAttnHead,
      activationsActivationIdxSelected,
    ],
    queryFn: () =>
      fetchActivationsValues(
        sampleWorkingSelected,
        activationSelected,
        selectedLayer,
        selectedAttnHead,
        activationsActivationIdxSelected as number
      ),
    enabled:
      modeVisualization === "activations" &&
      activationsTypeSelected === "value" &&
      activationSelected !== "" &&
      activationsActivationIdxSelected !== undefined,
  })

  const { data: dataActivationsSum } = useQuery({
    queryKey: [
      "dataActivationsSum",
      sampleWorkingSelected,
      activationSelected,
      selectedLayer,
      selectedAttnHead,
    ],
    queryFn: () =>
      fetchActivationsSum(
        sampleWorkingSelected,
        activationSelected,
        selectedLayer,
        selectedAttnHead
      ),
    enabled:
      modeVisualization === "activations" &&
      activationSelected !== "" &&
      activationsTypeSelected === "sum",
  })

  useEffect(() => {
    if (isFetchingForward) {
      setStartedFetchingForward(true)
    } else {
      if (startedFetchingForward) {
        setHasForwardData((currentList) => {
          const updatedList = [...currentList]
          updatedList[sampleWorkingSelected] = true
          return updatedList
        })
        setStartedFetchingForward(false)
      }
    }
  }, [isFetchingForward])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Control") {
        setIsCtrlPressed(true)
      }
    }
    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === "Control") {
        setIsCtrlPressed(false)
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    window.addEventListener("keyup", handleKeyUp)
    return () => {
      window.removeEventListener("keydown", handleKeyDown)
      window.removeEventListener("keyup", handleKeyUp)
    }
  }, [])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Alt") {
        setIsAltPressed(true)
      }
    }
    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === "Alt") {
        setIsAltPressed(false)
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    window.addEventListener("keyup", handleKeyUp)
    return () => {
      window.removeEventListener("keydown", handleKeyDown)
      window.removeEventListener("keyup", handleKeyUp)
    }
  }, [])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Meta") {
        setIsCommandPressed(true)
      }
    }
    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === "Meta") {
        setIsCommandPressed(false)
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    window.addEventListener("keyup", handleKeyUp)
    return () => {
      window.removeEventListener("keydown", handleKeyDown)
      window.removeEventListener("keyup", handleKeyUp)
    }
  }, [])

  const backgroundData = (index: number) => {
    switch (modeVisualization) {
      case "attention":
        if (selectedToken !== undefined) {
          if (attentionOpposite === false && dataAttention) {
            return dataAttention[index]
          } else if (attentionOpposite === true && dataAttentionOpposite) {
            return dataAttentionOpposite[index]
          } else {
            return null
          }
        } else {
          return null
        }
      case "loss":
        if (dataLoss) {
          return dataLoss[index]
        } else {
          return null
        }
      case "similarities":
        switch (similaritiesTypeSelected) {
          case "tokens":
            if (dataSimilaritiesTokens) {
              switch (normalizationMode) {
                case "default":
                  return dataSimilaritiesTokens.similarities[index]
                case "normalized":
                  console.log("normalized")
                  return dataSimilaritiesTokens.similarities_normalized[index]
                case "normalizedNoOutliers":
                  return dataSimilaritiesTokens
                    .similarities_normalized_no_outliers[index]
                default:
                  return null
              }
            } else {
              return null
            }
          case "previous":
            if (dataSimilaritiesPrevious) {
              switch (normalizationMode) {
                case "default":
                  return dataSimilaritiesPrevious.similarities[index]
                case "normalized":
                  return dataSimilaritiesPrevious.similarities_normalized[index]
                case "normalizedNoOutliers":
                  return dataSimilaritiesPrevious
                    .similarities_normalized_no_outliers[index]
                default:
                  return null
              }
            } else {
              return null
            }
          case "previousResidual":
            if (dataSimilaritiesPreviousResidual) {
              switch (normalizationMode) {
                case "default":
                  return dataSimilaritiesPreviousResidual.similarities[index]
                case "normalized":
                  return dataSimilaritiesPreviousResidual
                    .similarities_normalized[index]
                case "normalizedNoOutliers":
                  return dataSimilaritiesPreviousResidual
                    .similarities_normalized_no_outliers[index]
                default:
                  return null
              }
            } else {
              return null
            }
          default:
            return null
        }
      case "activations":
        if (activationsTypeSelected === "value") {
          if (dataActivationsValues) {
            switch (normalizationMode) {
              case "default":
                return dataActivationsValues.activations[index]
              case "normalized":
                return dataActivationsValues.activations_normalized[index]
              case "normalizedNoOutliers":
                return dataActivationsValues.activations_normalized_no_outliers[
                  index
                ]
              default:
                return null
            }
          }
          return null
        }
        if (activationsTypeSelected === "sum") {
          if (dataActivationsSum) {
            switch (normalizationMode) {
              case "default":
                return dataActivationsSum.activations_sum[index]
              case "normalized":
                return dataActivationsSum.activations_sum_normalized[index]
              case "normalizedNoOutliers":
                return dataActivationsSum
                  .activations_sum_normalized_no_outliers[index]
              default:
                return null
            }
          }
          return null
        }
        return null
      default:
        return null
    }
  }

  return (
    <div className="flex">
      <Menu />
      <div
        className={`p-3 mt-28 overflow-y-auto w-full ${
          showPredictions ? "h-[calc(100vh-255px)]" : "h-[calc(100vh-115px)]"
        }`}
      >
        {samplesWorkingTokens[sampleWorkingSelected] &&
          samplesWorkingTokens[sampleWorkingSelected].map((token, index) => (
            <TokenElement
              key={index}
              token={token}
              index={index}
              valueBackground={backgroundData(index)}
            />
          ))}
      </div>
      {showPredictions && <Predictions />}
    </div>
  )
}
