import type { ReactNode } from "react"
import {
  ArrowRight,
  Bot,
  ClipboardList,
  FileText,
  RadioTower,
  ShieldAlert,
} from "lucide-react"

import {
  DistrictSummary,
  EmergencyPriorityLocation,
  HighRiskLocation,
  LiveContextSummary,
  LiveDistrictContext,
  MonitoringSummary,
} from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ActiveView } from "@/components/dashboard/types"
import { CompactDriverList, DecisionMetric, RiskBadge } from "@/components/dashboard/shared"

const WORKFLOW = ["Brief", "Inspect", "Prioritize", "Simulate", "Report"]

export function CommandBriefing({
  summaries,
  priorityLocations,
  highRiskLocations,
  liveSummary,
  liveDistricts,
  liveContextError,
  monitoring,
  loading,
  onNavigate,
  onOpenLocation,
  onDistrictChange,
}: {
  summaries: DistrictSummary[]
  priorityLocations: EmergencyPriorityLocation[]
  highRiskLocations: HighRiskLocation[]
  liveSummary: LiveContextSummary | null
  liveDistricts: LiveDistrictContext[]
  liveContextError: string | null
  monitoring: MonitoringSummary | null
  loading: boolean
  onNavigate: (view: ActiveView) => void
  onOpenLocation: (recordId: string, district?: string) => void
  onDistrictChange: (district: string) => void
}) {
  const highestRiskDistrict = summaries[0]
  const totalPlaces = summaries.reduce(
    (total, summary) => total + summary.monitored_places,
    0
  )
  const highRiskCount = summaries.reduce(
    (total, summary) => total + summary.high_risk_count,
    0
  )
  const priorityCount = summaries.reduce(
    (total, summary) =>
      total + summary.critical_priority_count + summary.elevated_priority_count,
    0
  )
  const topRisk = highRiskLocations[0]
  const attentionArea = liveSummary?.highest_attention_area
  const boardDistricts = liveDistricts.length ? liveDistricts : liveSummary?.exposed_districts ?? []

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardTitle>Morning Briefing</CardTitle>
          <CardDescription>
            Start here before field teams are dispatched. Review attention areas,
            weather pressure, priority places, and the report handover path.
          </CardDescription>
          <CardAction>
            <LiveStatusBadge status={liveSummary?.status} error={liveContextError} />
          </CardAction>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="grid gap-2 md:grid-cols-5">
            {WORKFLOW.map((step, index) => (
              <button
                key={step}
                type="button"
                onClick={() => onNavigate(workflowTarget(step))}
                className="rounded-lg border border-border bg-muted/20 px-3 py-2 text-left text-sm transition-colors hover:bg-accent/40"
              >
                <div className="text-xs text-muted-foreground">Step {index + 1}</div>
                <div className="font-medium">{step}</div>
              </button>
            ))}
          </div>

          <div className="grid gap-3 lg:grid-cols-4">
            <DecisionMetric
              label="Highest attention area"
              value={attentionArea?.district ?? highestRiskDistrict?.district ?? "-"}
              detail={
                attentionArea
                  ? `${attentionArea.need_review_count} places need review`
                  : highestRiskDistrict
                    ? `${highestRiskDistrict.average_baseline_risk_score.toFixed(4)} average risk`
                    : "waiting for district data"
              }
              loading={loading}
            />
            <DecisionMetric
              label="Monitored nationally"
              value={totalPlaces.toLocaleString()}
              detail="places across provider scope"
              loading={loading}
            />
            <DecisionMetric
              label="Need action"
              value={priorityCount.toLocaleString()}
              detail="critical or elevated priority"
              loading={loading}
            />
            <DecisionMetric
              label="High-risk places"
              value={highRiskCount.toLocaleString()}
              detail="baseline exposure level"
              loading={loading}
            />
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 2xl:grid-cols-[1.1fr_0.9fr]">
        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShieldAlert className="size-4 text-cyan-300" />
                Attention Board
              </CardTitle>
              <CardDescription>
                Top districts by urgency, live rainfall pressure, and review load.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>District</TableHead>
                    <TableHead>Rain</TableHead>
                    <TableHead>River</TableHead>
                    <TableHead className="text-right">Need action</TableHead>
                    <TableHead>Reason</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {boardDistricts.slice(0, 6).map((district) => (
                    <TableRow
                      key={district.district}
                      className="cursor-pointer"
                      onClick={() => {
                        onDistrictChange(district.district)
                        onNavigate("priority-list")
                      }}
                    >
                      <TableCell>{district.district}</TableCell>
                      <TableCell>
                        <PressureBadge pressure={district.rainfall_pressure} />
                        <div className="mt-1 text-xs text-muted-foreground">
                          {formatRain(district.next_24h_rain_mm)} / 24h
                        </div>
                      </TableCell>
                      <TableCell>
                        <PressureBadge pressure={district.river_pressure} />
                      </TableCell>
                      <TableCell className="text-right">{district.need_review_count}</TableCell>
                      <TableCell className="max-w-60 truncate text-sm text-muted-foreground">
                        {district.top_reason}
                      </TableCell>
                    </TableRow>
                  ))}
                  {!boardDistricts.length ? (
                    <TableRow>
                      <TableCell colSpan={5} className="h-20 text-center text-muted-foreground">
                        Live district context is unavailable. Baseline risk remains available.
                      </TableCell>
                    </TableRow>
                  ) : null}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Immediate action queue</CardTitle>
              <CardDescription>First places to review before teams move.</CardDescription>
              <CardAction>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => onNavigate("priority-list")}
                >
                  Open queue <ArrowRight />
                </Button>
              </CardAction>
            </CardHeader>
            <CardContent className="grid gap-3">
              {priorityLocations.slice(0, 3).map((location) => (
                <button
                  key={location.record_id}
                  type="button"
                  onClick={() => onOpenLocation(location.record_id, location.district)}
                  className="rounded-lg border border-border p-3 text-left transition-colors hover:bg-accent/40"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="font-medium">{location.place_name}</div>
                      <div className="text-xs text-muted-foreground">
                        {location.district} / owner: district response desk
                      </div>
                    </div>
                    <RiskBadge level={location.baseline_risk_level} />
                  </div>
                  <div className="mt-3">
                    <CompactDriverList drivers={location.priority_reasons} />
                  </div>
                  <p className="mt-3 line-clamp-2 text-sm text-muted-foreground">
                    {location.recommended_action}
                  </p>
                </button>
              ))}
            </CardContent>
          </Card>
        </div>

        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Weather pressure</CardTitle>
              <CardDescription>
                Live provider context for the current highest attention area.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3">
              <PressureLine
                label="Rain next 24h"
                value={formatRain(liveSummary?.weather_pressure.next_24h_rain_mm)}
                pressure={liveSummary?.weather_pressure.rainfall_pressure}
              />
              <PressureLine
                label="Rain next 7d"
                value={formatRain(liveSummary?.weather_pressure.next_7d_rain_mm)}
                pressure={liveSummary?.weather_pressure.rainfall_pressure}
              />
              <PressureLine
                label="River signal"
                value={liveSummary?.weather_pressure.river_pressure ?? "Unavailable"}
                pressure={liveSummary?.weather_pressure.river_pressure}
              />
              <div className="rounded-lg border border-border p-3 text-sm text-muted-foreground">
                {liveSummary?.rainfall_outlook ??
                  "Live rainfall unavailable. Continue with baseline priority and retry live context before the demo."}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Report ready</CardTitle>
              <CardDescription>
                Convert the current evidence into a handover artifact.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3">
              <BriefingAction
                icon={<FileText />}
                label="Place action brief"
                detail={
                  topRisk
                    ? `Start from ${topRisk.place_name}, ${topRisk.district}`
                    : "Open a place and create a handover brief"
                }
                onClick={() => onNavigate("reports")}
              />
              <BriefingAction
                icon={<ClipboardList />}
                label="Guidance library"
                detail="Open SOPs, policies, and field evidence"
                onClick={() => onNavigate("response-documents")}
              />
              <BriefingAction
                icon={<Bot />}
                label="Draft with Copilot"
                detail="Turn priority evidence into a concise action brief"
                onClick={() => onNavigate("copilot")}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Review status</CardTitle>
              <CardDescription>Recent assessments completed in this workspace.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-2">
              <DecisionMetric
                label="Reviews logged"
                value={(monitoring?.total_predictions ?? 0).toLocaleString()}
                detail="places scored or assessed"
                loading={loading}
              />
              <DecisionMetric
                label="Latest review"
                value={monitoring?.latest_prediction_at ? "available" : "-"}
                detail={monitoring?.latest_prediction_at ?? "no review logged yet"}
                loading={loading}
              />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

function workflowTarget(step: string): ActiveView {
  if (step === "Inspect") return "risk-map"
  if (step === "Prioritize") return "priority-list"
  if (step === "Simulate") return "scenario"
  if (step === "Report") return "reports"
  return "briefing"
}

function LiveStatusBadge({
  status,
  error,
}: {
  status?: string
  error: string | null
}) {
  if (error) {
    return <Badge variant="outline">Live data unavailable</Badge>
  }
  return (
    <Badge variant={status === "live" ? "secondary" : "outline"} className="gap-1.5">
      <RadioTower className="size-3" />
      {status === "live" ? "Live context" : status === "partial" ? "Partial live context" : "Live data unavailable"}
    </Badge>
  )
}

function PressureLine({
  label,
  value,
  pressure,
}: {
  label: string
  value: string
  pressure?: string
}) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-border p-3">
      <div>
        <div className="text-sm font-medium">{label}</div>
        <div className="text-sm text-muted-foreground">{value}</div>
      </div>
      <PressureBadge pressure={pressure} />
    </div>
  )
}

function PressureBadge({ pressure }: { pressure?: string }) {
  return <Badge variant={pressure === "High" || pressure === "Severe" ? "destructive" : "outline"}>{pressure ?? "Unavailable"}</Badge>
}

function formatRain(value?: number | null) {
  return value == null ? "Unavailable" : `${value.toFixed(1)} mm`
}

function BriefingAction({
  icon,
  label,
  detail,
  onClick,
}: {
  icon: ReactNode
  label: string
  detail: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-3 rounded-lg border border-border p-3 text-left transition-colors hover:bg-accent/40"
    >
      <span className="text-cyan-300 [&_svg]:size-4">{icon}</span>
      <span className="min-w-0">
        <span className="block font-medium">{label}</span>
        <span className="block truncate text-sm text-muted-foreground">{detail}</span>
      </span>
    </button>
  )
}
