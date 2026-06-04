function Skeleton({ className }: { className?: string }) {
  return <div className={`animate-pulse bg-[rgba(255,255,255,0.06)] rounded-lg ${className}`} />
}

export default function DashboardLoading() {
  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-9 w-28" />
      </div>
      <div className="grid grid-cols-3 gap-4 mb-8">
        {[0, 1, 2].map(i => (
          <div key={i} className="surface-panel p-4">
            <Skeleton className="h-8 w-12 mb-2" />
            <Skeleton className="h-4 w-24" />
          </div>
        ))}
      </div>
      <div className="space-y-3">
        {[0, 1, 2].map(i => (
          <div key={i} className="surface-panel p-5">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <Skeleton className="h-5 w-48 mb-3" />
                <Skeleton className="h-4 w-32" />
              </div>
              <Skeleton className="h-6 w-16" />
            </div>
            <Skeleton className="h-1.5 w-full mt-3" />
          </div>
        ))}
      </div>
    </div>
  )
}
