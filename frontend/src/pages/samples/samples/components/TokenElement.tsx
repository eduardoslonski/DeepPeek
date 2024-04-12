import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  activationSelectedAtom,
  activationsHighlightDividerAtom,
  activationsHighlightLimitsAtom,
  activationsTypeSelectedAtom,
  altMultiplyFactorAttentionAtom,
  downloadNameAtom,
  hasForwardDataAtom,
  isAltPressedAtom,
  isCommandPressedAtom,
  isCtrlPressedAtom,
  modeVisualizationAtom,
  normalizationModeAtom,
  opacityVisualizationAtom,
  sampleWorkingSelectedAtom,
  selectedAttnHeadAtom,
  selectedLayerAtom,
  selectedTokensSamplesWorkingAtom,
  showLineBreakTokenAtom,
  showTooltipsAtom,
} from "@/lib/atoms"
import { useAtom, useAtomValue } from "jotai"
import {
  calculateColorActivationsSum,
  calculateColorActivationsValues,
  calculateColorAttention,
  calculateColorLoss,
  calculateColorSimilarities,
} from "../utilsTokenElement"
import { useQuery } from "@tanstack/react-query"
import { fetchActivationVector } from "@/api/activationVectorService"

interface TokenElementProps {
  token: string
  index: number
  valueBackground: number | null
}

export default function TokenElement({
  token,
  index,
  valueBackground,
}: TokenElementProps) {
  const showTooltips = useAtomValue(showTooltipsAtom)
  const showLineBreakToken = useAtomValue(showLineBreakTokenAtom)
  const isCtrlPressed = useAtomValue(isCtrlPressedAtom)
  const isAltPressed = useAtomValue(isAltPressedAtom)
  const isCommandPressed = useAtomValue(isCommandPressedAtom)
  const sampleWorkingSelected = useAtomValue(sampleWorkingSelectedAtom)
  const [selectedTokensSamplesWorking, setSelectedTokensSamplesWorking] =
    useAtom(selectedTokensSamplesWorkingAtom)
  const hasForwardData = useAtomValue(hasForwardDataAtom)
  const modeVisualization = useAtomValue(modeVisualizationAtom)
  const opacityVisualization = useAtomValue(opacityVisualizationAtom)
  const altMultiplyFactorAttention = useAtomValue(
    altMultiplyFactorAttentionAtom
  )
  const normalizationMode = useAtomValue(normalizationModeAtom)
  const selectedLayer = useAtomValue(selectedLayerAtom)
  const selectedAttnHead = useAtomValue(selectedAttnHeadAtom)
  const activationSelected = useAtomValue(activationSelectedAtom)
  const downloadName = useAtomValue(downloadNameAtom)
  const activationsTypeSelected = useAtomValue(activationsTypeSelectedAtom)
  const activationsHighlightLimits = useAtomValue(
    activationsHighlightLimitsAtom
  )
  const activationsHighlightDivider = useAtomValue(
    activationsHighlightDividerAtom
  )

  const getBackgroundColorToken = () => {
    if (hasForwardData[sampleWorkingSelected] === false) {
      return "white"
    } else if (
      isCtrlPressed &&
      selectedTokensSamplesWorking[sampleWorkingSelected] === index
    ) {
      return "rgb(125, 211, 252)"
    } else if (modeVisualization === "attention") {
      return calculateColorAttention(
        valueBackground as number,
        opacityVisualization,
        isAltPressed,
        altMultiplyFactorAttention
      )
    } else if (modeVisualization === "loss") {
      return calculateColorLoss(valueBackground as number, opacityVisualization)
    } else if (modeVisualization === "similarities") {
      return calculateColorSimilarities(
        valueBackground as number,
        opacityVisualization,
        normalizationMode
      )
    } else if (modeVisualization === "activations") {
      if (activationsTypeSelected === "sum") {
        return calculateColorActivationsSum(
          valueBackground as number,
          opacityVisualization,
          normalizationMode
        )
      } else {
        return calculateColorActivationsValues(
          valueBackground as number,
          opacityVisualization,
          activationsHighlightLimits,
          activationsHighlightDivider,
          normalizationMode
        )
      }
    }
    return "white"
  }

  const { refetch: refetchActivationVector } = useQuery({
    queryKey: [
      "dataActivationVector",
      sampleWorkingSelected,
      activationSelected,
      selectedLayer,
      selectedAttnHead,
      index,
    ],
    queryFn: () =>
      fetchActivationVector(
        sampleWorkingSelected,
        activationSelected,
        selectedLayer,
        selectedAttnHead,
        index
      ),
    enabled: false,
  })

  const activationsWithAttnHead = ["q", "k", "v", "o", "o_mm_dense"]

  const handleClickToken = async () => {
    if (isCommandPressed) {
      if (activationSelected) {
        const { data } = await refetchActivationVector()

        if (data) {
          const floatArray = new Float32Array(data)

          const blob = new Blob([floatArray], {
            type: "application/octet-stream",
          })

          const url = URL.createObjectURL(blob)

          const a = document.createElement("a")
          a.href = url
          a.download = `${
            downloadName ? `${downloadName}_` : ""
          }${index}_${activationSelected}_layer_${selectedLayer}${
            activationsWithAttnHead.includes(activationSelected)
              ? `_attn_head_${selectedAttnHead}`
              : ""
          }.bin`
          document.body.appendChild(a)

          a.click()

          document.body.removeChild(a)
          URL.revokeObjectURL(url)
        }
      }
    } else {
      setSelectedTokensSamplesWorking((prevList) =>
        prevList.map((item, idx) =>
          idx === sampleWorkingSelected
            ? item === index
              ? undefined
              : index
            : item
        )
      )
    }
  }

  const spanElement = () => {
    return (
      <span
        className={`whitespace-pre-wrap ${
          hasForwardData[sampleWorkingSelected]
            ? "cursor-pointer hover:brightness-90"
            : ""
        } ${
          hasForwardData[sampleWorkingSelected] === false
            ? "text-gray-300 opacity-100"
            : ""
        } ${token.includes("\n") ? "text-gray-300" : ""}${
          token === "<|endoftext|>" &&
          !isCtrlPressed &&
          hasForwardData[sampleWorkingSelected] === true
            ? "text-sky-700"
            : ""
        } ${
          selectedTokensSamplesWorking[sampleWorkingSelected] === index
            ? "[text-shadow:_0_0_1px_rgb(0_0_0_/_100%)]"
            : ""
        }`}
        style={{
          backgroundColor: getBackgroundColorToken(),
        }}
        onClick={() =>
          hasForwardData[sampleWorkingSelected] === true && handleClickToken()
        }
      >
        {token.includes("\n") && showLineBreakToken
          ? token.replace(/\n/g, "\\n")
          : token}
      </span>
    )
  }

  return (
    <>
      {showTooltips ? (
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>{spanElement()}</TooltipTrigger>
            <TooltipContent>
              <span>{valueBackground?.toFixed(3)}</span>
              <span className="text-gray-400">{` ${index}`}</span>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ) : (
        spanElement()
      )}
      {showLineBreakToken &&
        token
          .split("\n")
          .map((_, index, array) =>
            index < array.length - 1 ? <br key={index} /> : null
          )}
    </>
  )
}
