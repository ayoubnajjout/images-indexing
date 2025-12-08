import React from 'react';

const SkeletonLoader = ({ className = '', variant = 'rectangle' }) => {
  const variants = {
    rectangle: 'w-full h-48',
    circle: 'w-12 h-12 rounded-full',
    text: 'w-full h-4',
    card: 'w-full h-64',
  };
  
  return (
    <div className={`bg-slate-200 animate-pulse ${variants[variant]} ${className}`} />
  );
};

export const ImageCardSkeleton = () => (
  <div className="card p-4">
    <SkeletonLoader variant="rectangle" className="mb-3 rounded" />
    <SkeletonLoader variant="text" className="mb-2" />
    <SkeletonLoader variant="text" className="w-2/3" />
  </div>
);

export const ListSkeleton = ({ items = 3 }) => (
  <div className="space-y-3">
    {Array.from({ length: items }).map((_, i) => (
      <div key={i} className="flex items-center gap-3">
        <SkeletonLoader variant="circle" />
        <div className="flex-1 space-y-2">
          <SkeletonLoader variant="text" />
          <SkeletonLoader variant="text" className="w-3/4" />
        </div>
      </div>
    ))}
  </div>
);

export default SkeletonLoader;
