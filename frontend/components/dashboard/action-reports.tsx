import type { ReactNode } from "react"
import { Bot, FileText, MapPinned, SlidersHorizontal } from "lucide-react"

import {
  DistrictSummary,
  EmergencyPriorityLocation,
  LiveContextSummary,
  LiveDistrictContext,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { ActiveView } from "@/components/dashboard/types"
import { CompactDriverList } from "@/components/dashboard/shared"

export function ActionReportsPanel({
  summaries,
  priorityLocations,
  liveSummary,
  liveDistricts,
  onNavigate,
  onOpenLocation,
  onDraftCopilot,
}: {
  summaries: DistrictSummary[]
  priorityLocations: EmergencyPriorityLocation[]
  liveSummary: LiveContextSummary | null
  liveDistricts: LiveDistrictContext[]
  onNavigate: (view: ActiveView) => void
  onOpenLocation: (recordId: string, district?: string) => void
  onDraftCopilot: (prompt: string) => void
}) {
  const topDistrict = summaries[0]
  const topPlace = priorityLocations[0]
  const topLiveDistrict = liveSummary?.highest_attention_area ?? liveDistricts[0]

  return (
    <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
      <Card>
        <CardHeader>
          <CardTitle>Action Reports</CardTitle>
          <CardDescription>
            Choose the handover format for a place, district, or scenario.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3">
          <ReportAction
            icon={<MapPinned />}
            title="Location action brief"
            detail={
              topPlace
                ? `Contains risk, reason, live weather, owner, and recommended action for ${topPlace.place_name}.`
                : "Contains risk, reason, live weather, owner, and recommended action."
            }
            action="Create"
            onClick={() =>
              topPlace
                ? onOpenLocation(topPlace.record_id, topPlace.district)
                : onNavigate("risk-map")
            }
          />
          <ReportAction
            icon={<SlidersHorizontal />}
            title="Scenario handover report"
            detail="Contains baseline versus scenario risk, changed assumptions, and action change."
            action="Simulate"
            onClick={() => onNavigate("scenario")}
          />
          <ReportAction
            icon={<Bot />}
            title="District action brief"
            detail={
              topLiveDistrict
                ? `Contains live rainfall context and priority queue for ${topLiveDistrict.district}.`
                : "Contains live rainfall context and priority queue for the highest attention area."
            }
            action="Draft"
            onClick={() => {
              const district = topLiveDistrict?.district ?? topDistrict?.district ?? "the highest-risk district"
              onDraftCopilot(
                `Create a concise action brief for ${district}. Include priority places, main risk reasons, recommended actions, and evidence sources.`
              )
            }}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Report starting point</CardTitle>
          <CardDescription>
            First report to create for the current morning briefing.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {topPlace ? (
            <div className="rounded-lg border border-border p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-lg font-semibold">{topPlace.place_name}</div>
                  <div className="font-mono text-xs text-muted-foreground">
                    {topPlace.district} / {topPlace.record_id}
                  </div>
                </div>
                <div className="rounded-lg border border-border px-3 py-2 text-right">
                  <div className="text-xs text-muted-foreground">Priority</div>
                  <div className="font-medium">{topPlace.operational_priority}</div>
                </div>
              </div>
              <div className="mt-4">
                <div className="mb-2 text-xs text-muted-foreground">Reason</div>
                <CompactDriverList drivers={topPlace.priority_reasons} />
              </div>
              <div className="mt-4 rounded-lg border border-border p-3 text-sm">
                <div className="mb-1 text-xs text-muted-foreground">Recommended action</div>
                <p>{topPlace.recommended_action}</p>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                <Button
                  type="button"
                  onClick={() => onOpenLocation(topPlace.record_id, topPlace.district)}
                >
                  <MapPinned /> Open in Risk Map
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => onNavigate("scenario")}
                >
                  <FileText /> Build scenario PDF
                </Button>
              </div>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-border p-5 text-sm text-muted-foreground">
              No reports created yet. Open Risk Map or Scenario Lab to create the first report.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function ReportAction({
  icon,
  title,
  detail,
  action,
  onClick,
}: {
  icon: ReactNode
  title: string
  detail: string
  action: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center justify-between gap-4 rounded-lg border border-border p-4 text-left transition-colors hover:bg-accent/40"
    >
      <span className="flex min-w-0 items-center gap-3">
        <span className="text-cyan-300 [&_svg]:size-4">{icon}</span>
        <span className="min-w-0">
          <span className="block font-medium">{title}</span>
          <span className="block truncate text-sm text-muted-foreground">{detail}</span>
        </span>
      </span>
      <span className="shrink-0 text-sm text-muted-foreground">{action}</span>
    </button>
  )
}
