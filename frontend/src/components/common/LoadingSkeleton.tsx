export function LoadingSkeleton() {
  return (
    <div className="animate-pulse space-y-6">
      <div className="flex gap-6">
        <div className="h-72 flex-1 rounded-lg bg-gray-800" />
        <div className="h-72 flex-2 rounded-lg bg-gray-800" />
      </div>
      <div className="h-10 w-2/3 rounded bg-gray-800" />
      <div className="h-64 rounded-lg bg-gray-800" />
    </div>
  );
}
