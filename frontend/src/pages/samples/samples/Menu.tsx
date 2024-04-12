import { Toggle } from "@/components/ui/toggle"
import SampleSelector from "./components/SampleSelector"
import SampleWorkingToggleGroup from "./components/SampleWorkingToggleGroup"
import {
  activationSelectedAtom,
  activationsActivationIdxSelectedAtom,
  activationsTypeSelectedAtom,
  modeVisualizationAtom,
  showPredictionsAtom,
  similaritiesTypeSelectedAtom,
} from "@/lib/atoms"
import { useAtom, useSetAtom } from "jotai"
import { Separator } from "@/components/ui/separator"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import Options from "./components/Options"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Input } from "@/components/ui/input"

export default function Menu() {
  const [showPredictions, setShowPredictions] = useAtom(showPredictionsAtom)
  const [modeVisualization, setModeVisualization] = useAtom(
    modeVisualizationAtom
  )
  const [activationSelected, setActivationSelected] = useAtom(
    activationSelectedAtom
  )

  const [similaritiesTypeSelected, setSimilaritiesTypeSelected] = useAtom(
    similaritiesTypeSelectedAtom
  )
  const [activationsTypeSelected, setActivationsTypeSelected] = useAtom(
    activationsTypeSelectedAtom
  )

  const setActivationsActivationIdxSelected = useSetAtom(
    activationsActivationIdxSelectedAtom
  )

  interface ToggleItemHighlightProps {
    item: string
  }

  function ToggleItemHighlight({ item }: ToggleItemHighlightProps) {
    const item_lower = item.toLowerCase()
    return (
      <ToggleGroupItem
        size={"sm"}
        onClick={() => setModeVisualization(item_lower)}
        variant={"outline"}
        className="rounded-full min-w-[60px] h-8"
        value={item_lower}
        aria-label={`Toggle ${item_lower}`}
      >
        {item}
      </ToggleGroupItem>
    )
  }

  const togglesMenu = ["Attention", "Loss", "Similarities", "Activations"]

  const activationsToSelectAll = [
    "input",
    "input_layernorm",
    "q",
    "k",
    "v",
    "o",
    "o_mm_dense",
    "dense_attention",
    "post_attention_layernorm",
    "mlp_h_to_4",
    "mlp_4_to_h",
    "output",
  ]

  const activationsToSelectResidual = [
    "dense_attention",
    "dense_attention_residual",
    "mlp_4_to_h",
    "output",
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

  const activationsToSelect =
    modeVisualization === "similarities" &&
    similaritiesTypeSelected === "previousResidual"
      ? activationsToSelectResidual
      : activationsToSelectAll

  return (
    <div className="border-l-4 border-muted fixed flex flex-col gap-4 z-40 bg-white border-b-2 border-muted w-4/6 p-3 h-28">
      <div className="flex gap-2 justify-between items-center">
        <div className="flex gap-2">
          <SampleSelector />
          <Toggle
            pressed={showPredictions}
            size={"sm"}
            variant={"selecting"}
            className="px-2 h-8"
            onClick={() => setShowPredictions(!showPredictions)}
          >
            Predictions
          </Toggle>
          <Separator orientation="vertical" className="h-8 w-0.5" />
          <ToggleGroup
            className="gap-2"
            defaultValue={modeVisualization}
            type={"single"}
            variant={"selectingOutline"}
            onValueChange={(value) => {
              !value && setModeVisualization(value)
            }}
          >
            {togglesMenu.map((item) => {
              return <ToggleItemHighlight key={item} item={item} />
            })}
          </ToggleGroup>
          <div className="flex justify-start items-center gap-2">
            <Select
              value={activationSelected}
              onValueChange={(value) => setActivationSelected(value)}
            >
              <SelectTrigger className="w-[110px] h-7">
                <SelectValue placeholder="Activation" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectLabel>Activation</SelectLabel>
                  {activationsToSelect.map((item) => {
                    return (
                      <SelectItem key={item} value={item}>
                        {item}
                      </SelectItem>
                    )
                  })}
                </SelectGroup>
              </SelectContent>
            </Select>

            {(modeVisualization === "similarities" ||
              modeVisualization === "activations") && (
              <div className="flex items-center bg-white border-input border rounded-full h-7">
                <ToggleGroup
                  type={"single"}
                  className="gap-0"
                  value={
                    modeVisualization === "similarities"
                      ? similaritiesTypeSelected
                      : activationsTypeSelected
                  }
                  onValueChange={(value) => {
                    value &&
                      (modeVisualization === "similarities"
                        ? setSimilaritiesTypeSelected(value)
                        : setActivationsTypeSelected(value))
                  }}
                >
                  <ToggleGroupItem
                    className="rounded-full border-0"
                    size={"xs"}
                    variant={"outline"}
                    value={
                      modeVisualization === "similarities" ? "tokens" : "sum"
                    }
                  >
                    {modeVisualization === "similarities" ? "Tokens" : "Sum"}
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    className="rounded-full border-0"
                    size={"xs"}
                    variant={"outline"}
                    value={
                      modeVisualization === "similarities"
                        ? "previous"
                        : "value"
                    }
                  >
                    {modeVisualization === "similarities"
                      ? "Previous"
                      : "Value"}
                  </ToggleGroupItem>
                  {modeVisualization === "similarities" && (
                    <ToggleGroupItem
                      className="rounded-full border-0"
                      size={"xs"}
                      variant={"outline"}
                      value="previousResidual"
                    >
                      Prev Residual
                    </ToggleGroupItem>
                  )}
                </ToggleGroup>
              </div>
            )}
            {modeVisualization === "activations" &&
              activationsTypeSelected === "value" && (
                <Input
                  className="w-24 h-7"
                  placeholder="activation"
                  onKeyDown={handleKeyDownInputActivation}
                  type="number"
                />
              )}
          </div>
        </div>
        <Options />
      </div>
      <div className="flex inline-block gap-2">
        <SampleWorkingToggleGroup />
      </div>
    </div>
  )
}
