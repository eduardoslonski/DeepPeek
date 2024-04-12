import { fetchSampleText } from "@/api/sampleTextService"
import {
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogFooter,
} from "@/components/ui/alert-dialog"
import { Textarea } from "@/components/ui/textarea"
import {
  hasForwardDataAtom,
  sampleWorkingSelectedAtom,
  samplesWorkingTextAtom,
  samplesWorkingTokensAtom,
} from "@/lib/atoms"
import { useQuery } from "@tanstack/react-query"
import { useAtomValue, useSetAtom } from "jotai"
import { useEffect, useState } from "react"

interface SampleDialogProps {
  dataset?: string
  sampleIdx?: number
  setSheetOpen: (isOpen: boolean) => void
}

export default function SampleDialog({
  dataset,
  sampleIdx,
  setSheetOpen,
}: SampleDialogProps) {
  const { data: dataSampleText } = useQuery({
    queryKey: ["fetchSampleText"],
    queryFn: () => fetchSampleText(dataset as string, sampleIdx as number),
    enabled: dataset !== undefined && sampleIdx !== undefined,
  })

  const [textAreaValue, setTextAreaValue] = useState("")

  useEffect(() => {
    if (dataSampleText && dataset !== undefined) {
      setTextAreaValue(dataSampleText.text)
    }
  }, [dataSampleText])

  const sampleWorkingSelected = useAtomValue(sampleWorkingSelectedAtom)
  const samplesWorkingText = useAtomValue(samplesWorkingTextAtom)
  const setHasForwardData = useSetAtom(hasForwardDataAtom)
  const setSamplesWorkingText = useSetAtom(samplesWorkingTextAtom)

  if (
    dataset === undefined &&
    samplesWorkingText[sampleWorkingSelected] !== undefined &&
    textAreaValue === ""
  ) {
    setTextAreaValue(samplesWorkingText[sampleWorkingSelected])
  }
  return (
    <>
      <Textarea
        value={textAreaValue}
        onChange={(event) => setTextAreaValue(event.target.value)}
        className="h-96 text-base min-h-[600px]"
      ></Textarea>
      <AlertDialogFooter>
        <AlertDialogCancel>Cancel</AlertDialogCancel>
        <AlertDialogAction
          onClick={() => {
            if (textAreaValue !== "") {
              setSamplesWorkingText((currentList) => {
                const updatedList = [...currentList]
                if (sampleWorkingSelected < updatedList.length) {
                  updatedList[sampleWorkingSelected] = textAreaValue
                } else {
                  updatedList[sampleWorkingSelected] = textAreaValue
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
              setSheetOpen(false)
            }
          }}
        >
          Continue
        </AlertDialogAction>
      </AlertDialogFooter>
    </>
  )
}
