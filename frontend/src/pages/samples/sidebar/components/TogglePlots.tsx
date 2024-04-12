import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { PrimitiveAtom, useAtom } from "jotai"

interface TogglePlotsProps {
  items: string[]
  atomPlots: PrimitiveAtom<string[]>
}

interface ToggleItemPlotsProps {
  item: string
}

export default function TogglePlots({ items, atomPlots }: TogglePlotsProps) {
  const [selectedPlotsSidebar, setSelectectedPlotsSidebar] = useAtom(atomPlots)

  const updateSelectedPlots = (item: string) => {
    setSelectectedPlotsSidebar((prevSelectedPlots) => {
      if (prevSelectedPlots.includes(item)) {
        return prevSelectedPlots.filter((plot) => plot !== item)
      } else {
        return [...prevSelectedPlots, item]
      }
    })
  }

  function ToggleItemPlots({ item }: ToggleItemPlotsProps) {
    return (
      <ToggleGroupItem
        size={"xs"}
        onClick={() => updateSelectedPlots(item)}
        variant={"selecting"}
        className="rounded-full min-w-[60px]"
        value={item}
        aria-label={`Toggle ${item}`}
      >
        {item}
      </ToggleGroupItem>
    )
  }

  return (
    <div>
      <ToggleGroup
        type="multiple"
        value={selectedPlotsSidebar}
        className="flex flex-wrap justify-start gap-2"
      >
        {items.map((item) => (
          <ToggleItemPlots key={item} item={item} />
        ))}
      </ToggleGroup>
    </div>
  )
}
