import { atom } from "jotai"

// Model config
interface ModelConfigType {
  name: string | undefined
  dModel: number | undefined
  nLayers: number | undefined
  nAttnHeads: number | undefined
  dAttnHead: number | undefined
}

export const modelConfigAtom = atom<ModelConfigType>({
  name: undefined,
  dModel: undefined,
  nLayers: undefined,
  nAttnHeads: undefined,
  dAttnHead: undefined,
})

// Menu
interface SampleToSelectType {
  title: string
  tokens: number
}

interface CategorySamplesToSelectType {
  category: string
  items: SampleToSelectType[]
}
export const samplesToSelectAtom = atom<CategorySamplesToSelectType[]>([])
export const samplesWorkingMenuAtom = atom<number>(1)

// Menu
export const showPredictionsAtom = atom<boolean>(false)
export const modeVisualizationAtom = atom<string>("")

export const activationSelectedAtom = atom<string>("")
export const similaritiesTypeSelectedAtom = atom<string>("tokens")

export const activationsTypeSelectedAtom = atom<string>("sum")
export const activationsActivationIdxSelectedAtom = atom<number | undefined>(
  undefined
)
export const attentionOppositeAtom = atom<boolean>(false)

export const downloadNameAtom = atom<string>("")

// Samples
export const samplesWorkingTextAtom = atom<string[]>([])
export const samplesWorkingTokensAtom = atom<string[][]>([])
export const sampleWorkingSelectedAtom = atom<number>(0)

export const hasForwardDataAtom = atom<boolean[]>([false])
export const isFetchingForwardAtom = atom<boolean>(false)

// Data
export const selectedLayerAtom = atom<number>(0)
export const selectedAttnHeadAtom = atom<number>(0)
export const selectedTokensSamplesWorkingAtom = atom<(number | undefined)[]>([
  undefined,
])
export const ropeModeAtom = atom<string>("full")

// UI
export const showTooltipsAtom = atom<boolean>(false)
export const showLineBreakTokenAtom = atom<boolean>(true)

export const isCtrlPressedAtom = atom<boolean>(false)
export const isAltPressedAtom = atom<boolean>(false)
export const isCommandPressedAtom = atom<boolean>(false)

export const opacityVisualizationAtom = atom<number>(0.5)
export const altMultiplyFactorAttentionAtom = atom<number>(20)
export const normalizationModeAtom = atom<string>("default")
export const activationsHighlightLimitsAtom = atom<[number, number]>([-3, 3])
export const activationsHighlightDividerAtom = atom<number>(0)

// Sidebar
export const showScatterPlotsAtom = atom<boolean>(false)

export const selectedHistogramsSidebarAtom = atom<string[]>([])
export const selectedScattersSidebarAtom = atom<string[]>([])

// // Histogram
export const binsCurrentAtom = atom<number[]>([])
export const histogramCurrentAtom = atom<number[]>([])
