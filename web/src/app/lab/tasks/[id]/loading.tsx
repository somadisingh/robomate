function Skeleton({ className }: { className?: string }) {
  return <div className={`animate-pulse bg-[rgba(255,255,255,0.06)] rounded-lg ${className}`} />
}

export default function TaskLoading() {
  return (
    <div>
      <Skeleton className="h-4 w-24 mb-6" />
      <div className="surface-panel p-6 mb-6">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div className="flex-1">
            <Skeleton className="h-7 w-64 mb-2" />
            <div className="flex gap-2">
              <Skeleton className="h-5 w-16" />
              <Skeleton className="h-5 w-20" />
            </div>
          </div>
          <div className="text-right">
            <Skeleton className="h-8 w-16 mb-1" />
            <Skeleton className="h-4 w-24" />
          </div>
        </div>
        <Skeleton className="h-4 w-full mb-1.5" />
        <Skeleton className="h-2 w-full mt-3" />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {[0, 1, 2].map(i => (
          <div key={i} className="surface-panel overflow-hidden">
            <Skeleton className="w-full aspect-video rounded-none" />
            <div className="p-3">
              <div className="flex justify-between items-start">
                <Skeleton className="h-4 w-16" />
                <div className="flex gap-1.5">
                  <Skeleton className="h-6 w-12" />
                  <Skeleton className="h-6 w-16" />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
