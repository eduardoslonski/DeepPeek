import { Button } from "@/components/ui/button"
import { PrimitiveAtom, useAtom } from "jotai"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { useState } from "react"
import { Input } from "../ui/input"

interface PreviousNextProps {
  name?: string
  atom: PrimitiveAtom<number>
  minValue?: number
  maxValue: number
}

export default function PreviousNext({
  name,
  atom,
  minValue = 0,
  maxValue,
}: PreviousNextProps) {
  const [currentValue, setCurrentValue] = useAtom(atom)
  const [inputValue, setInputValue] = useState<number>(currentValue)

  const handleButtonsChangeLayer = async (amount: number) => {
    const newValue = currentValue + amount
    if (newValue < (minValue ? minValue : 0) || newValue > maxValue) {
      return
    }
    setCurrentValue(newValue)
    setInputValue(newValue)
  }

  const handleKeyDown = async (
    event: React.KeyboardEvent<HTMLInputElement>
  ) => {
    if (event.key == "Enter") {
      if (inputValue > maxValue || inputValue < minValue) {
        setInputValue(currentValue)
        return
      }
      setCurrentValue(inputValue)
    }
  }

  return (
    <div className="flex items-center">
      {name && <span className="mr-2">{name}</span>}
      <Button
        variant="outline"
        size="icon"
        className="h-7 w-7"
        onClick={() => handleButtonsChangeLayer(-1)}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>
      <Input
        type="number"
        className="h-7 border-none mx-2 text-center p-0.5"
        style={{ width: `${maxValue.toString().length * 10 + 10}px` }}
        onBlur={() => setInputValue(currentValue)}
        onChange={(e) => setInputValue(parseInt(e.target.value))}
        onKeyDown={handleKeyDown}
        value={inputValue}
      />
      <Button
        variant="outline"
        size="icon"
        className="h-7 w-7"
        onClick={() => handleButtonsChangeLayer(+1)}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>
    </div>
  )
}
