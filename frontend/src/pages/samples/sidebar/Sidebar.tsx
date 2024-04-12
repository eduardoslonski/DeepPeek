import PreviousNext from "@/components/custom/previousNext"
import {
  modelConfigAtom,
  selectedAttnHeadAtom,
  selectedHistogramsSidebarAtom,
  selectedLayerAtom,
  selectedScattersSidebarAtom,
  showScatterPlotsAtom,
} from "@/lib/atoms"
import { useQuery } from "@tanstack/react-query"
import { useAtom, useAtomValue } from "jotai"
import { useEffect } from "react"
import TogglePlots from "./components/TogglePlots"
import { fetchConfig } from "@/api/configService"
import HistogramGroup from "./components/HistogramGroup"
import { Toggle } from "@/components/ui/toggle"
import ScatterGroup from "./components/ScatterGroup"

export default function Sidebar() {
  const [modelConfig, setModelConfig] = useAtom(modelConfigAtom)
  const selectedHistogramsSidebar = useAtomValue(selectedHistogramsSidebarAtom)
  const selectedScattersSidebar = useAtomValue(selectedScattersSidebarAtom)
  const [showScatterPlots, setShowScatterPlots] = useAtom(showScatterPlotsAtom)

  const { data: dataModelConfig, isSuccess } = useQuery({
    queryKey: ["fetchConfig"],
    queryFn: fetchConfig,
    enabled: true,
  })

  useEffect(() => {
    if (isSuccess) {
      setModelConfig({
        name: dataModelConfig.model_name,
        dModel: dataModelConfig.d_model,
        nLayers: dataModelConfig.n_layers,
        nAttnHeads: dataModelConfig.n_attn_heads,
        dAttnHead: dataModelConfig.d_attn_head,
      })
    }
  }, [isSuccess])

  const itemsPlots = [
    "input",
    "input_layernorm",
    "q",
    "k",
    "v",
    "q_rope",
    "k_rope",
    "o",
    "o_mm_dense",
    "dense_attention",
    "post_attention_layernorm",
    "mlp_h_to_4",
    "mlp_4_to_h",
    "output",
  ]

  return (
    <>
      <div>
        <div className="border-l-4 border-muted fixed flex justify-between items-center z-40 bg-white border-b-2 border-muted w-2/6 p-4 h-16">
          <PreviousNext
            name="Layer"
            atom={selectedLayerAtom}
            maxValue={modelConfig?.nLayers ? modelConfig?.nLayers - 1 : 0}
          />
          <Toggle
            pressed={showScatterPlots}
            size={"xs"}
            variant={"selecting"}
            className="px-2"
            onClick={() => setShowScatterPlots(!showScatterPlots)}
          >
            Scatter Plots
          </Toggle>
          <PreviousNext
            name="Head"
            atom={selectedAttnHeadAtom}
            maxValue={modelConfig?.nAttnHeads ? modelConfig?.nAttnHeads - 1 : 0}
          />
        </div>
      </div>
      <div className="p-4 mt-16 border-l-4 border-muted h-full overflow-y-auto h-[calc(100vh-71px)] flex flex-col gap-8">
        {showScatterPlots && (
          <div>
            <TogglePlots
              items={itemsPlots}
              atomPlots={selectedScattersSidebarAtom}
            ></TogglePlots>
            <div className="mt-4">
              {selectedScattersSidebar.map((item, idx) => (
                <ScatterGroup key={idx} typeActivation={item} />
              ))}
            </div>
          </div>
        )}
        <div>
          <TogglePlots
            items={itemsPlots}
            atomPlots={selectedHistogramsSidebarAtom}
          ></TogglePlots>
          <div className="mt-4">
            {selectedHistogramsSidebar.map((item, idx) => (
              <HistogramGroup key={idx} type={item} />
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
