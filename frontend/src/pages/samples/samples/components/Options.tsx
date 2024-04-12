import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import {
  activationsHighlightDividerAtom,
  activationsHighlightLimitsAtom,
  activationsTypeSelectedAtom,
  altMultiplyFactorAttentionAtom,
  attentionOppositeAtom,
  downloadNameAtom,
  modeVisualizationAtom,
  normalizationModeAtom,
  opacityVisualizationAtom,
  ropeModeAtom,
  showLineBreakTokenAtom,
  showTooltipsAtom,
} from "@/lib/atoms"
import { useAtom, useAtomValue } from "jotai"
import { useState } from "react"

export default function Options() {
  const [opacityVisualization, setOpacityVisualization] = useAtom(
    opacityVisualizationAtom
  )
  const [altMultiplyFactorAttention, setAltMultiplyFactorAttention] = useAtom(
    altMultiplyFactorAttentionAtom
  )
  const [attentionOpposite, setAttentionOpposite] = useAtom(
    attentionOppositeAtom
  )
  const [showTooltips, setShowTooltips] = useAtom(showTooltipsAtom)
  const [showLineBreakToken, setShowLineBreakToken] = useAtom(
    showLineBreakTokenAtom
  )
  const [ropeMode, setRopeMode] = useAtom(ropeModeAtom)
  const [normalizationMode, setNormalizationMode] = useAtom(
    normalizationModeAtom
  )
  const modeVisualization = useAtomValue(modeVisualizationAtom)
  const [downloadName, setDownloadName] = useAtom(downloadNameAtom)

  const [textInputDownloadName, setTextInputDownloadName] =
    useState(downloadName)

  const activationsTypeSelected = useAtomValue(activationsTypeSelectedAtom)

  const [activationsHighlightLimits, setActivationsHighlightLimits] = useAtom(
    activationsHighlightLimitsAtom
  )

  const [activationsHighlightDivider, setActivationsHighlightDivider] = useAtom(
    activationsHighlightDividerAtom
  )

  const [
    textInputActivationsHighlightLimitsMin,
    setTextInputActivationsHighlightLimitsMin,
  ] = useState(activationsHighlightLimits[0])

  const [
    textInputActivationsHighlightLimitsMax,
    setTextInputActivationsHighlightLimitsMax,
  ] = useState(activationsHighlightLimits[1])

  const [
    textInputActivationsHighlightDivider,
    setTextInputActivationsHighlightDivider,
  ] = useState(activationsHighlightDivider)

  function handleLimitsChange(value: string, type: string) {
    const numericValue = parseFloat(value)
    if (!isNaN(numericValue)) {
      if (type === "min") {
        setActivationsHighlightLimits((prevArray: [number, number]) => [
          numericValue,
          prevArray[1],
        ])
      } else if (type === "max") {
        setActivationsHighlightLimits((prevArray: [number, number]) => [
          prevArray[0],
          numericValue,
        ])
      }
    }
  }

  function handleDividerChange(value: string) {
    const numericValue = parseFloat(value)
    if (!isNaN(numericValue)) {
      setActivationsHighlightDivider(numericValue)
    }
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="secondary" className="p-2 h-7">
          Options
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="flex flex-col p-4 gap-6">
        <div className="flex flex-col gap-3">
          <p className="text-sm">Opacity</p>
          <Slider
            value={[opacityVisualization]}
            max={1}
            step={0.1}
            onValueChange={(value) => setOpacityVisualization(value[0])}
          />
        </div>
        <div className="flex flex-col gap-3">
          <p className="text-sm">Tooltips</p>
          <Switch
            checked={showTooltips}
            onCheckedChange={(value) => setShowTooltips(value)}
          />
        </div>
        <div className="flex flex-col gap-3">
          <p className="text-sm">Show \n token</p>
          <Switch
            checked={showLineBreakToken}
            onCheckedChange={(value) => setShowLineBreakToken(value)}
          />
        </div>
        {modeVisualization === "attention" && (
          <>
            <div className="flex flex-col gap-3">
              <p className="text-sm">[Alt] Multiply Factor</p>
              <Slider
                value={[altMultiplyFactorAttention]}
                min={1}
                max={40}
                step={1}
                onValueChange={(value) =>
                  setAltMultiplyFactorAttention(value[0])
                }
              />
            </div>
            <div className="flex flex-col gap-3">
              <p className="text-sm">Attention Opposite</p>
              <Switch
                checked={attentionOpposite}
                onCheckedChange={(value) => setAttentionOpposite(value)}
              />
            </div>
          </>
        )}
        {modeVisualization === "similarities" && (
          <div className="flex flex-col gap-3">
            <p className="text-sm">RoPE</p>
            <RadioGroup
              value={ropeMode}
              onValueChange={(value) => setRopeMode(value)}
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="full" id="rope_r1" />
                <Label htmlFor="rope_r1">Full</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="applied" id="rope_r2" />
                <Label htmlFor="rope_r2">Applied</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="not_applied" id="rope_r3" />
                <Label htmlFor="rope_r3">Not applied</Label>
              </div>
            </RadioGroup>
          </div>
        )}
        {(modeVisualization === "similarities" ||
          modeVisualization === "activations") && (
          <div className="flex flex-col gap-4">
            <p className="text-sm">Normalization</p>
            <RadioGroup
              value={normalizationMode}
              onValueChange={(value) => setNormalizationMode(value)}
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="default" id="normalization_r1" />
                <Label htmlFor="normalization_r1">Default</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="normalized" id="normalization_r2" />
                <Label htmlFor="normalization_r2">Normalized</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem
                  value="normalizedNoOutliers"
                  id="normalization_r3"
                />
                <Label htmlFor="normalization_r3">Normalized no outliers</Label>
              </div>
            </RadioGroup>
          </div>
        )}
        {modeVisualization === "activations" &&
          activationsTypeSelected === "value" && (
            <div>
              <p className="text-sm">Limits Activation Highlight</p>
              <div className="flex gap-2 justify-between">
                <div>
                  <p className="text-xs">Min</p>
                  <Input
                    className={`w-16 h-7 ${
                      textInputActivationsHighlightLimitsMin ===
                      activationsHighlightLimits[0]
                        ? "text-black"
                        : "text-gray-500"
                    }`}
                    defaultValue={activationsHighlightLimits[0]}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        handleLimitsChange(event.currentTarget.value, "min")
                      }
                    }}
                    onBlur={(event) =>
                      handleLimitsChange(event.currentTarget.value, "min")
                    }
                    onChange={(event) =>
                      setTextInputActivationsHighlightLimitsMin(
                        parseFloat(event.target.value)
                      )
                    }
                  />
                </div>
                <div>
                  <p className="text-xs">Divider</p>
                  <Input
                    className={`w-16 h-7 ${
                      textInputActivationsHighlightDivider ===
                      activationsHighlightDivider
                        ? "text-black"
                        : "text-gray-500"
                    }`}
                    defaultValue={activationsHighlightDivider}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        handleDividerChange(event.currentTarget.value)
                      }
                    }}
                    onBlur={(event) => handleDividerChange(event.target.value)}
                    onChange={(event) =>
                      setTextInputActivationsHighlightDivider(
                        parseFloat(event.target.value)
                      )
                    }
                  />
                </div>
                <div>
                  <p className="text-xs">Max</p>
                  <Input
                    className={`w-16 h-7 ${
                      textInputActivationsHighlightLimitsMax ===
                      activationsHighlightLimits[1]
                        ? "text-black"
                        : "text-gray-500"
                    }`}
                    defaultValue={activationsHighlightLimits[1]}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        handleLimitsChange(event.currentTarget.value, "max")
                      }
                    }}
                    onBlur={(event) =>
                      handleLimitsChange(event.currentTarget.value, "max")
                    }
                    onChange={(event) =>
                      setTextInputActivationsHighlightLimitsMax(
                        parseFloat(event.target.value)
                      )
                    }
                  />
                </div>
              </div>
            </div>
          )}
        <div className="flex flex-col gap-3">
          <p className="text-sm">Download name</p>
          <Input
            className={`w-32 h-7 ${
              textInputDownloadName === downloadName
                ? "text-black"
                : "text-gray-500"
            }`}
            defaultValue={downloadName}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                setDownloadName(event.currentTarget.value)
              }
            }}
            onChange={(event) => setTextInputDownloadName(event.target.value)}
          />
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
