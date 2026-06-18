"use client"

import { FormEvent, ReactNode, useEffect, useMemo, useState } from "react"
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  DatabaseZap,
  Gauge,
  Loader2,
  Play,
  RadioTower,
  ShieldCheck,
  Waves,
} from "lucide-react"

import {
  getHealth,
  getModelInfo,
  getMonitoringSummary,
  ModelInfo,
  MonitoringSummary,
  predictFloodRisk,
  PredictionResult,
} from "@/lib/api"
import { sampleRecord } from "@/lib/sample-record"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
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
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"

type ApiState = "checking" | "online" | "offline"

const initialPayload = JSON.stringify(sampleRecord, null, 2)

export default function Home() {
  const [apiState, setApiState] = useState<ApiState>("checking")
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [monitoring, setMonitoring] = useState<MonitoringSummary | null>(null)
  const [payload, setPayload] = useState(initialPayload)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function fetchDashboardData() {
    const [health, model, summary] = await Promise.all([
      getHealth(),
      getModelInfo(),
      getMonitoringSummary(),
    ])
    return { health, model, summary }
  }

  useEffect(() => {
    let ignore = false

    async function loadDashboard() {
      try {
        const { health, model, summary } = await fetchDashboardData()
        if (ignore) return
        setApiState(health.model_loaded ? "online" : "offline")
        setModelInfo(model)
        setMonitoring(summary)
      } catch (err: unknown) {
        if (ignore) return
        setApiState("offline")
        setError(err instanceof Error ? err.message : "Backend is unavailable")
      } finally {
        if (!ignore) setLoading(false)
      }
    }

    void loadDashboard()

    return () => {
      ignore = true
    }
  }, [])

  async function handlePredict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setPredicting(true)
    setError(null)

    try {
      const record = JSON.parse(payload) as Record<string, unknown>
      const result = await predictFloodRisk(record)
      setPrediction(result)
      setMonitoring(await getMonitoringSummary())
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Prediction request failed")
    } finally {
      setPredicting(false)
    }
  }

  const riskTotal = useMemo(
    () =>
      (monitoring?.low_risk_count ?? 0) +
      (monitoring?.medium_risk_count ?? 0) +
      (monitoring?.high_risk_count ?? 0),
    [monitoring]
  )

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 border-b border-border pb-5 lg:flex-row lg:items-center lg:justify-between">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="gap-1.5 border-cyan-400/30 text-cyan-300">
                <Waves className="size-3" />
                FloodLens
              </Badge>
              <Badge
                variant={apiState === "online" ? "secondary" : "destructive"}
                className="gap-1.5"
              >
                <RadioTower className="size-3" />
                API {apiState}
              </Badge>
              {modelInfo ? (
                <Badge variant="outline" className="font-mono">
                  {modelInfo.model_version}
                </Badge>
              ) : null}
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
                Flood Risk Intelligence Dashboard
              </h1>
              <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
                Prediction serving, model governance, and operational monitoring for Sri Lanka flood risk.
              </p>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm sm:flex">
            <StatusPill label="Model" value={modelInfo ? "loaded" : "pending"} />
            <StatusPill label="Logs" value={`${monitoring?.total_predictions ?? 0} events`} />
          </div>
        </header>

        {error ? (
          <Alert variant="destructive">
            <AlertTriangle className="size-4" />
            <AlertTitle>Backend request failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
          <MetricCard
            icon={<BrainCircuit />}
            label="Model version"
            value={modelInfo?.model_version ?? "—"}
            detail={`${modelInfo?.feature_count ?? "—"} features`}
            loading={loading}
          />
          <MetricCard
            icon={<Gauge />}
            label="OOF MAE"
            value={formatNumber(modelInfo?.metrics.oof_mae, 6)}
            detail={`RMSE ${formatNumber(modelInfo?.metrics.oof_rmse, 6)}`}
            loading={loading}
          />
          <MetricCard
            icon={<DatabaseZap />}
            label="Predictions"
            value={monitoring?.total_predictions ?? 0}
            detail="successful API events"
            loading={loading}
          />
          <MetricCard
            icon={<Activity />}
            label="Average risk"
            value={formatNullable(monitoring?.average_risk_score)}
            detail="from logged predictions"
            loading={loading}
          />
          <MetricCard
            icon={<AlertTriangle />}
            label="High risk"
            value={monitoring?.high_risk_count ?? 0}
            detail="logged locations"
            loading={loading}
          />
          <MetricCard
            icon={<ShieldCheck />}
            label="Latest event"
            value={formatTime(monitoring?.latest_prediction_at)}
            detail="prediction log"
            loading={loading}
          />
        </section>

        <Tabs defaultValue="overview" className="gap-4">
          <TabsList className="w-full justify-start overflow-x-auto sm:w-fit">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="prediction">Prediction</TabsTrigger>
            <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="grid gap-4 xl:grid-cols-[1.25fr_0.75fr]">
            <Card>
              <CardHeader>
                <CardTitle>Model serving status</CardTitle>
                <CardDescription>Active artifact metadata from the FastAPI backend.</CardDescription>
                <CardAction>
                  <Badge variant="secondary">{apiState}</Badge>
                </CardAction>
              </CardHeader>
              <CardContent>
                {loading || !modelInfo ? (
                  <InfoSkeleton />
                ) : (
                  <div className="grid gap-3 sm:grid-cols-2">
                    <InfoRow label="Version" value={modelInfo.model_version} />
                    <InfoRow label="Trained" value={formatDate(modelInfo.trained_at)} />
                    <InfoRow label="Rows" value={modelInfo.metrics.n_train.toLocaleString()} />
                    <InfoRow label="Features" value={modelInfo.feature_count.toString()} />
                    <InfoRow label="Test std" value={formatNumber(modelInfo.metrics.test_std, 4)} />
                    <InfoRow label="Training time" value={`${modelInfo.metrics.training_time_s}s`} />
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Risk distribution</CardTitle>
                <CardDescription>Counts from logged prediction events.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {loading || !monitoring ? (
                  <InfoSkeleton />
                ) : (
                  <>
                    <RiskBar
                      label="Low"
                      value={monitoring.low_risk_count}
                      total={riskTotal}
                      className="bg-emerald-400"
                    />
                    <RiskBar
                      label="Medium"
                      value={monitoring.medium_risk_count}
                      total={riskTotal}
                      className="bg-amber-400"
                    />
                    <RiskBar
                      label="High"
                      value={monitoring.high_risk_count}
                      total={riskTotal}
                      className="bg-rose-400"
                    />
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="prediction" className="grid gap-4 xl:grid-cols-[1fr_0.7fr]">
            <Card>
              <CardHeader>
                <CardTitle>Single location prediction</CardTitle>
                <CardDescription>Submit a full test-row style record to the deployed model API.</CardDescription>
              </CardHeader>
              <CardContent>
                <form className="space-y-3" onSubmit={handlePredict}>
                  <div className="space-y-2">
                    <Label htmlFor="prediction-payload">Prediction payload</Label>
                    <Textarea
                      id="prediction-payload"
                      value={payload}
                      onChange={(event) => setPayload(event.target.value)}
                      className="min-h-[430px] resize-y font-mono text-xs leading-5"
                      spellCheck={false}
                    />
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button type="submit" disabled={predicting}>
                      {predicting ? <Loader2 className="animate-spin" /> : <Play />}
                      Run prediction
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setPayload(initialPayload)}
                    >
                      Reset sample
                    </Button>
                  </div>
                </form>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Prediction result</CardTitle>
                <CardDescription>Latest response from `POST /predict`.</CardDescription>
              </CardHeader>
              <CardContent>
                {prediction ? (
                  <div className="space-y-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="font-mono text-sm text-muted-foreground">
                          {prediction.record_id}
                        </p>
                        <p className="mt-2 text-5xl font-semibold tracking-tight">
                          {prediction.flood_risk_score.toFixed(6)}
                        </p>
                      </div>
                      <RiskBadge level={prediction.risk_level} />
                    </div>
                    <Separator />
                    <div className="grid gap-3 text-sm">
                      <InfoRow label="Model" value={prediction.model_version} />
                      <InfoRow label="Risk level" value={prediction.risk_level} />
                      <InfoRow label="Monitoring" value="event logged" />
                    </div>
                  </div>
                ) : (
                  <div className="rounded-lg border border-dashed border-border p-5 text-sm text-muted-foreground">
                    No prediction submitted in this browser session.
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="monitoring" className="grid gap-4 xl:grid-cols-[0.8fr_1.2fr]">
            <Card>
              <CardHeader>
                <CardTitle>Model versions</CardTitle>
                <CardDescription>Version usage from prediction logs.</CardDescription>
              </CardHeader>
              <CardContent>
                {monitoring && Object.keys(monitoring.model_versions).length > 0 ? (
                  <div className="space-y-3">
                    {Object.entries(monitoring.model_versions).map(([version, count]) => (
                      <div key={version} className="flex items-center justify-between gap-3">
                        <span className="font-mono text-sm">{version}</span>
                        <Badge variant="outline">{count}</Badge>
                      </div>
                    ))}
                  </div>
                ) : (
                  <EmptyLine />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Top districts by predictions</CardTitle>
                <CardDescription>Operational activity from logged API calls.</CardDescription>
                <CardAction>
                  <BarChart3 className="size-4 text-muted-foreground" />
                </CardAction>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>District</TableHead>
                      <TableHead className="text-right">Predictions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {monitoring?.top_districts_by_predictions.length ? (
                      monitoring.top_districts_by_predictions.map((district) => (
                        <TableRow key={district.district}>
                          <TableCell>{district.district}</TableCell>
                          <TableCell className="text-right font-mono">
                            {district.count}
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={2} className="text-muted-foreground">
                          No district activity logged yet.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </main>
  )
}

function MetricCard({
  icon,
  label,
  value,
  detail,
  loading,
}: {
  icon: ReactNode
  label: string
  value: ReactNode
  detail: string
  loading: boolean
}) {
  return (
    <Card size="sm" className="min-h-32">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-muted-foreground">
          <span className="text-cyan-300 [&_svg]:size-4">{icon}</span>
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-8 w-24" />
        ) : (
          <div className="truncate text-2xl font-semibold tracking-tight">{value}</div>
        )}
        <p className="mt-2 text-xs text-muted-foreground">{detail}</p>
      </CardContent>
    </Card>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg border border-border px-3 py-2">
      <span className="text-muted-foreground">{label}</span>
      <span className="truncate text-right font-mono text-sm">{value}</span>
    </div>
  )
}

function RiskBar({
  label,
  value,
  total,
  className,
}: {
  label: string
  value: number
  total: number
  className: string
}) {
  const percent = total > 0 ? Math.round((value / total) * 100) : 0

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span>{label}</span>
        <span className="font-mono text-muted-foreground">
          {value} / {percent}%
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div className={className} style={{ width: `${percent}%`, height: "100%" }} />
      </div>
    </div>
  )
}

function RiskBadge({ level }: { level: PredictionResult["risk_level"] }) {
  const className =
    level === "High"
      ? "border-rose-400/30 bg-rose-400/10 text-rose-300"
      : level === "Medium"
        ? "border-amber-400/30 bg-amber-400/10 text-amber-300"
        : "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"

  return (
    <Badge variant="outline" className={className}>
      {level}
    </Badge>
  )
}

function StatusPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border px-3 py-2">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-sm">{value}</div>
    </div>
  )
}

function InfoSkeleton() {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {Array.from({ length: 6 }).map((_, index) => (
        <Skeleton key={index} className="h-9 rounded-lg" />
      ))}
    </div>
  )
}

function EmptyLine() {
  return (
    <div className="rounded-lg border border-dashed border-border p-5 text-sm text-muted-foreground">
      No prediction events logged yet.
    </div>
  )
}

function formatNumber(value: number | undefined, digits: number) {
  return typeof value === "number" ? value.toFixed(digits) : "—"
}

function formatNullable(value: number | null | undefined) {
  return typeof value === "number" ? value.toFixed(6) : "—"
}

function formatDate(value: string | null | undefined) {
  if (!value) return "—"
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value))
}

function formatTime(value: string | null | undefined) {
  if (!value) return "—"
  return new Intl.DateTimeFormat("en", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value))
}
