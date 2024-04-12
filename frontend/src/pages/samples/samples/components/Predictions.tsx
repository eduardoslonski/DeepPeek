import { fetchPredictions } from "@/api/predictionsService"
import {
  sampleWorkingSelectedAtom,
  samplesWorkingTokensAtom,
  selectedTokensSamplesWorkingAtom,
} from "@/lib/atoms"
import { useQuery } from "@tanstack/react-query"
import { useAtomValue } from "jotai"

export default function Predictions() {
  const selectedTokensSamplesWorking = useAtomValue(
    selectedTokensSamplesWorkingAtom
  )
  const sampleWorkingSelected = useAtomValue(sampleWorkingSelectedAtom)
  const samplesWorkingTokens = useAtomValue(samplesWorkingTokensAtom)

  const selectedToken = selectedTokensSamplesWorking[sampleWorkingSelected]

  const { data: dataPredictions } = useQuery({
    queryKey: ["dataPredictions", sampleWorkingSelected],
    queryFn: () => fetchPredictions(sampleWorkingSelected, 5),
    enabled: selectedToken !== undefined,
  })

  return (
    <div className="fixed flex flex-col p-3 bottom-0 w-4/6 h-36 bg-white border-t-2 border-muted">
      {dataPredictions &&
        selectedToken !== undefined &&
        dataPredictions.tokens[selectedToken as number].map(
          (token: string, index: number) => {
            const value = dataPredictions.values[selectedToken as number][index]
            const width = 700 * value
            return (
              <div key={index} className="flex items-center gap-1">
                <span className="w-28">{token === "\n" ? "\\n" : token}</span>
                <span className="text-muted-foreground w-12 text-right opacity-50">
                  {(value * 100).toFixed(0)}
                </span>
                <div
                  className={`h-4 ${
                    samplesWorkingTokens[sampleWorkingSelected][
                      selectedToken + 1
                    ] === token
                      ? "bg-blue-500"
                      : "bg-red-500"
                  }`}
                  style={{ width: `${width}px` }}
                ></div>
              </div>
            )
          }
        )}
    </div>
  )
}
