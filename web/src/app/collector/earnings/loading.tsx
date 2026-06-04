function Skeleton({ className }: { className?: string }) {
  return <div className={`animate-pulse bg-gray-100 rounded-lg ${className}`} />
}

export default function EarningsLoading() {
  return (
    <div>
      <Skeleton className="h-8 w-28 mb-8" />
      <div className="grid grid-cols-2 gap-4 mb-8">
        {[0, 1].map(i => (
          <div key={i} className="bg-white rounded-xl border border-gray-100 p-5">
            <Skeleton className="h-9 w-20 mb-2" />
            <Skeleton className="h-4 w-24" />
          </div>
        ))}
      </div>
      <div className="space-y-2">
        {[0, 1, 2].map(i => (
          <div key={i} className="bg-white rounded-xl border border-gray-100 p-4 flex items-center justify-between">
            <div>
              <Skeleton className="h-4 w-40 mb-1.5" />
              <Skeleton className="h-3 w-24" />
            </div>
            <div className="text-right">
              <Skeleton className="h-5 w-12 mb-1" />
              <Skeleton className="h-3 w-14" />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
