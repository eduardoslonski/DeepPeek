import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import SamplesPage from "./pages/samples/SamplesPage"

const queryClient = new QueryClient({})

function App() {
  return (
    <>
      <QueryClientProvider client={queryClient}>
        <SamplesPage />
      </QueryClientProvider>
    </>
  )
}

export default App
