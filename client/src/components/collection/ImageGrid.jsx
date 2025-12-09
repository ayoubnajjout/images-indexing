import React from 'react';
import ImageCard from './ImageCard';
import { ImageCardSkeleton } from '../ui/SkeletonLoader';
import Spinner from '../ui/Spinner';

const ImageGrid = ({ images, loading, onDelete, onDetect, lastImageRef, loadingMore }) => {
  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {Array.from({ length: 8 }).map((_, index) => (
          <ImageCardSkeleton key={index} />
        ))}
      </div>
    );
  }
  
  if (!images || images.length === 0) {
    return (
      <div className="text-center py-16">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-slate-100 rounded-full mb-4">
          <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-slate-900 mb-2">No images found</h3>
        <p className="text-slate-500">Upload some images to get started</p>
      </div>
    );
  }
  
  return (
    <>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {images.map((image, index) => {
          // Attach ref to the last image for pagination intersection observer
          if (index === images.length - 1 && lastImageRef) {
            return (
              <div ref={lastImageRef} key={image.id}>
                <ImageCard
                  image={image}
                  onDelete={onDelete}
                  onDetect={onDetect}
                />
              </div>
            );
          }
          return (
            <ImageCard
              key={image.id}
              image={image}
              onDelete={onDelete}
              onDetect={onDetect}
            />
          );
        })}
      </div>
      {loadingMore && (
        <div className="flex justify-center items-center py-8">
          <Spinner size="md" />
          <span className="ml-3 text-slate-600">Loading more images...</span>
        </div>
      )}
    </>
  );
};

export default ImageGrid;
