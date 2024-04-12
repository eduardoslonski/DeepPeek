import { Button } from "@/components/ui/button"
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import {
  hasForwardDataAtom,
  sampleWorkingSelectedAtom,
  samplesWorkingMenuAtom,
  selectedTokensSamplesWorkingAtom,
} from "@/lib/atoms"
import { useAtom, useSetAtom } from "jotai"

export default function SampleWorkingToggleGroup() {
  const [samplesWorkingMenu, setSamplesWorkingMenu] = useAtom(
    samplesWorkingMenuAtom
  )
  const [sampleWorkingSelected, setSampleWorkingSelected] = useAtom(
    sampleWorkingSelectedAtom
  )
  const setHasForwardData = useSetAtom(hasForwardDataAtom)
  const setSelectedTokensSamplesWorking = useSetAtom(
    selectedTokensSamplesWorkingAtom
  )

  return (
    <div className="flex inline-block gap-2">
      <ToggleGroup
        className="gap-2"
        type={"single"}
        value={String(sampleWorkingSelected)}
      >
        {Array.from({ length: samplesWorkingMenu }, (_, i) => i).map(
          (index) => (
            <ContextMenu key={index}>
              <ContextMenuTrigger>
                <ToggleGroupItem
                  size={"xs"}
                  onClick={() => setSampleWorkingSelected(index)}
                  variant={"outline"}
                  className="rounded-full min-w-[60px]"
                  value={String(index)}
                  key={index}
                >
                  {index}
                </ToggleGroupItem>
              </ContextMenuTrigger>
              <ContextMenuContent className="rounded-full mt-0 p-0">
                <ContextMenuItem className="text-sm flex gap-2 rounded-full">
                  <span className="text-red-600 font-semibold">X</span>
                  <span>Remove</span>
                </ContextMenuItem>
              </ContextMenuContent>
            </ContextMenu>
          )
        )}
      </ToggleGroup>
      <Button
        className="h-7 w-7 rounded-full"
        variant={"outline"}
        onClick={() => {
          setSamplesWorkingMenu((currentNumber) => currentNumber + 1)
          setHasForwardData((prevList) => [...prevList, false])
          setSelectedTokensSamplesWorking((prevList) => [
            ...prevList,
            undefined,
          ])
          setSampleWorkingSelected(samplesWorkingMenu)
        }}
      >
        +
      </Button>
    </div>
  )
}
