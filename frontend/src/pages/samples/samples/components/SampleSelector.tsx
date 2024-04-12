import { fetchSamplesToSelect } from "@/api/samplesToSelectService"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import {
  sampleWorkingSelectedAtom,
  samplesToSelectAtom,
  samplesWorkingTextAtom,
  samplesWorkingTokensAtom,
} from "@/lib/atoms"
import { useQuery } from "@tanstack/react-query"
import { useAtom, useAtomValue } from "jotai"
import { useEffect, useState } from "react"
import SampleDialog from "./SampleDialog"
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"

export default function SampleSelector() {
  const [samplesToSelect, setSamplesToSelect] = useAtom(samplesToSelectAtom)
  const [isSheetOpen, setSheetOpen] = useState(false)
  const samplesWorkingText = useAtomValue(samplesWorkingTextAtom)
  const samplesWorkingTokens = useAtomValue(samplesWorkingTokensAtom)
  const sampleWorkingSelected = useAtomValue(sampleWorkingSelectedAtom)

  const { data: dataSamplesToSelect } = useQuery({
    queryKey: ["fetchSamplesToSelect"],
    queryFn: () => fetchSamplesToSelect(),
  })

  useEffect(() => {
    if (dataSamplesToSelect) {
      setSamplesToSelect(dataSamplesToSelect)
    }
  }, [dataSamplesToSelect])

  return (
    <Sheet open={isSheetOpen} onOpenChange={setSheetOpen}>
      <SheetTrigger asChild>
        <Button size={"sm"} variant={"outline"} className="h-8">
          Samples
        </Button>
      </SheetTrigger>
      <SheetContent side={"left"} className="max-h-full overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Samples</SheetTitle>
        </SheetHeader>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <div className="border-2 border-muted p-3 rounded-md mt-3">
              {samplesWorkingTokens[sampleWorkingSelected] !== undefined ? (
                <>
                  <p className="font-semibold">Current Sample</p>
                  <div
                    className="flex flex-col cursor-pointer bg-white hover:bg-secondary text-start p-2 rounded hover:bg-secondary"
                    role="button"
                  >
                    <span>
                      {samplesWorkingText[sampleWorkingSelected].length > 64
                        ? samplesWorkingText[sampleWorkingSelected].substring(
                            0,
                            64
                          ) + "..."
                        : samplesWorkingText[sampleWorkingSelected]}
                    </span>
                    <span className="text-muted-foreground">
                      {samplesWorkingTokens[sampleWorkingSelected].length}
                    </span>
                  </div>
                </>
              ) : (
                <div
                  className="flex flex-col cursor-pointer bg-white hover:bg-secondary text-start p-2 rounded hover:bg-secondary"
                  role="button"
                >
                  <p className="text-muted-foreground">Write sample...</p>
                </div>
              )}
            </div>
          </AlertDialogTrigger>
          <AlertDialogContent className="min-w-[80%] min-h-[80%] flex flex-col gap-4">
            <SampleDialog setSheetOpen={setSheetOpen} />
          </AlertDialogContent>
        </AlertDialog>
        <Accordion type="multiple">
          {samplesToSelect.map((categorySamplesData) => (
            <AccordionItem
              key={categorySamplesData.category}
              value={categorySamplesData.category}
            >
              <AccordionTrigger>
                {categorySamplesData.category}
              </AccordionTrigger>
              <AccordionContent>
                {categorySamplesData.items.map((sample, index) => (
                  <AlertDialog key={index}>
                    <AlertDialogTrigger asChild>
                      <div
                        className="flex flex-col cursor-pointer bg-white hover:bg-secondary text-start p-2 rounded"
                        role="button"
                        tabIndex={0}
                      >
                        <span
                          className={
                            !sample.tokens ? `text-muted-foreground` : ""
                          }
                        >
                          {sample.title}
                        </span>
                        {sample.tokens && (
                          <span className="text-muted-foreground">
                            {sample.tokens}
                          </span>
                        )}
                      </div>
                    </AlertDialogTrigger>
                    <AlertDialogContent className="min-w-[80%] min-h-[80%] flex flex-col gap-4">
                      <SampleDialog
                        dataset={categorySamplesData.category}
                        sampleIdx={index}
                        setSheetOpen={setSheetOpen}
                      />
                    </AlertDialogContent>
                  </AlertDialog>
                ))}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </SheetContent>
    </Sheet>
  )
}
