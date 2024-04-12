import Sample from "./samples/Sample"
import Sidebar from "./sidebar/Sidebar"

export default function SamplesPage() {
  return (
    <div className="flex">
      <div className="w-4/6">
        <Sample />
      </div>
      <div className="w-2/6">
        <Sidebar />
      </div>
    </div>
  )
}
