"use client"

import { useMemo, useState } from "react"
import { MapPinned } from "lucide-react"

import { EmergencyPriorityLocation, LiveDistrictContext } from "@/lib/api"
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
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ALL_DISTRICTS } from "@/components/dashboard/types"
import { CompactDriverList, RiskBadge } from "@/components/dashboard/shared"

const REVIEW_STATUSES = [
  "Not reviewed",
  "Reviewing",
  "Action planned",
  "Report shared",
] as const

type ReviewStatus = (typeof REVIEW_STATUSES)[number]

export function BusinessPriorityList({
  district,
  districts,
  priorityLocations,
  liveDistricts,
  loading,
  onDistrictChange,
  onOpenLocation,
}: {
  district: string
  districts: string[]
  priorityLocations: EmergencyPriorityLocation[]
  liveDistricts: LiveDistrictContext[]
  loading: boolean
  onDistrictChange: (district: string) => void
  onOpenLocation: (recordId: string, district?: string) => void
}) {
  const [priorityFilter, setPriorityFilter] = useState("all")
  const [rainFilter, setRainFilter] = useState("all")
  const [statusFilter, setStatusFilter] = useState("all")
  const [reviewStatus, setReviewStatus] = useState<Record<string, ReviewStatus>>({})
  const liveByDistrict = useMemo(
    () => Object.fromEntries(liveDistricts.map((item) => [item.district, item])),
    [liveDistricts]
  )
  const filteredLocations = priorityLocations.filter((location) => {
    const status = reviewStatus[location.record_id] ?? "Not reviewed"
    const live = liveByDistrict[location.district]
    return (
      (priorityFilter === "all" || location.operational_priority === priorityFilter) &&
      (statusFilter === "all" || status === statusFilter) &&
      (rainFilter === "all" || live?.rainfall_pressure === rainFilter)
    )
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>Priority List</CardTitle>
        <CardDescription>
          Planner queue for places needing attention, action ownership, and report handover.
        </CardDescription>
        <CardAction className="flex flex-wrap gap-2">
          <Select
            value={district}
            onValueChange={(value) => {
              if (value) onDistrictChange(value)
            }}
          >
            <SelectTrigger className="w-52">
              <SelectValue placeholder="All districts" />
            </SelectTrigger>
            <SelectContent align="end">
              <SelectItem value={ALL_DISTRICTS}>All districts</SelectItem>
              {districts.map((item) => (
                <SelectItem key={item} value={item}>
                  {item}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select
            value={priorityFilter}
            onValueChange={(value) => {
              if (value) setPriorityFilter(value)
            }}
          >
            <SelectTrigger className="w-44">
              <SelectValue placeholder="Priority" />
            </SelectTrigger>
            <SelectContent align="end">
              <SelectItem value="all">All priorities</SelectItem>
              <SelectItem value="Critical">Critical</SelectItem>
              <SelectItem value="Elevated">Elevated</SelectItem>
              <SelectItem value="Watch">Watch</SelectItem>
              <SelectItem value="Routine">Routine</SelectItem>
            </SelectContent>
          </Select>
          <Select
            value={rainFilter}
            onValueChange={(value) => {
              if (value) setRainFilter(value)
            }}
          >
            <SelectTrigger className="w-44">
              <SelectValue placeholder="Rain pressure" />
            </SelectTrigger>
            <SelectContent align="end">
              <SelectItem value="all">All rain</SelectItem>
              <SelectItem value="Severe">Severe</SelectItem>
              <SelectItem value="High">High</SelectItem>
              <SelectItem value="Watch">Watch</SelectItem>
              <SelectItem value="Normal">Normal</SelectItem>
              <SelectItem value="Unavailable">Unavailable</SelectItem>
            </SelectContent>
          </Select>
          <Select
            value={statusFilter}
            onValueChange={(value) => {
              if (value) setStatusFilter(value)
            }}
          >
            <SelectTrigger className="w-44">
              <SelectValue placeholder="Review status" />
            </SelectTrigger>
            <SelectContent align="end">
              <SelectItem value="all">All status</SelectItem>
              {REVIEW_STATUSES.map((status) => (
                <SelectItem key={status} value={status}>
                  {status}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardAction>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[720px] rounded-lg border border-border">
          <Table>
            <TableHeader className="sticky top-0 z-10 bg-card">
              <TableRow>
                <TableHead className="w-16">Rank</TableHead>
                <TableHead className="min-w-56">Place</TableHead>
                <TableHead className="w-28">Priority</TableHead>
                <TableHead className="w-32">Live rain</TableHead>
                <TableHead className="min-w-56">Reason</TableHead>
                <TableHead className="min-w-80">Recommended action</TableHead>
                <TableHead className="w-44">Status</TableHead>
                <TableHead className="text-right">Open</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                Array.from({ length: 10 }).map((_, index) => (
                  <TableRow key={index}>
                    <TableCell colSpan={8}>
                      <Skeleton className="h-7 w-full" />
                    </TableCell>
                  </TableRow>
                ))
              ) : filteredLocations.length ? (
                filteredLocations.map((location) => {
                  const status = reviewStatus[location.record_id] ?? "Not reviewed"
                  const live = liveByDistrict[location.district]
                  return (
                    <TableRow key={location.record_id}>
                      <TableCell className="font-mono text-xs">#{location.rank}</TableCell>
                      <TableCell>
                        <div className="font-medium">{location.place_name}</div>
                        <div className="text-xs text-muted-foreground">
                          {location.district} / owner: district response desk
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-col gap-1">
                          <RiskBadge level={location.baseline_risk_level} />
                          <Badge variant="outline">{location.operational_priority}</Badge>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant={live?.rainfall_pressure === "High" || live?.rainfall_pressure === "Severe" ? "destructive" : "outline"}>
                          {live?.rainfall_pressure ?? "Unavailable"}
                        </Badge>
                        <div className="mt-1 text-xs text-muted-foreground">
                          {formatRain(live?.next_24h_rain_mm)} / 24h
                        </div>
                      </TableCell>
                      <TableCell>
                        <CompactDriverList drivers={location.priority_reasons} />
                      </TableCell>
                      <TableCell className="max-w-md text-sm text-muted-foreground">
                        {location.recommended_action}
                      </TableCell>
                      <TableCell>
                        <Select
                          value={status}
                          onValueChange={(value) => {
                            if (!value) return
                            setReviewStatus((current) => ({
                              ...current,
                              [location.record_id]: value as ReviewStatus,
                            }))
                          }}
                        >
                          <SelectTrigger className="w-40">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent align="end">
                            {REVIEW_STATUSES.map((item) => (
                              <SelectItem key={item} value={item}>
                                {item}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() =>
                            onOpenLocation(location.record_id, location.district)
                          }
                        >
                          <MapPinned /> Open
                        </Button>
                      </TableCell>
                    </TableRow>
                  )
                })
              ) : (
                <TableRow>
                  <TableCell colSpan={8} className="h-24 text-center text-muted-foreground">
                    No priority places match the current filters.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

function formatRain(value?: number | null) {
  return value == null ? "Unavailable" : `${value.toFixed(1)} mm`
}
