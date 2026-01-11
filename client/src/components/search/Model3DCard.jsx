/**
 * 3D Model Card Component
 * Displays a 3D model thumbnail with info and actions
 */

import { useState } from 'react';
import { model3DService } from '../../services';

const Model3DCard = ({ 
  model, 
  onSelect, 
  onSearch,
  selected = false,
  showSimilarity = false,
  compact = false
}) => {
  const [imageError, setImageError] = useState(false);
  const thumbnailUrl = model3DService.getThumbnailUrl(model.thumbnailUrl);
  
  const handleClick = () => {
    if (onSelect) {
      onSelect(model);
    }
  };

  const handleSearch = (e) => {
    e.stopPropagation();
    if (onSearch) {
      onSearch(model);
    }
  };

  return (
    <div
      onClick={handleClick}
      className={`
        relative bg-white rounded-lg shadow-md overflow-hidden cursor-pointer
        transition-all duration-200 hover:shadow-lg hover:scale-[1.02]
        ${selected ? 'ring-2 ring-blue-500' : ''}
        ${compact ? 'p-2' : 'p-3'}
      `}
    >
      {/* Thumbnail */}
      <div className={`relative ${compact ? 'h-24' : 'h-32'} bg-gray-100 rounded overflow-hidden mb-2`}>
        {thumbnailUrl && !imageError ? (
          <img
            src={thumbnailUrl}
            alt={model.name || model.filename}
            className="w-full h-full object-contain"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-400">
            <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} 
                d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>
          </div>
        )}
        
        {/* Similarity badge */}
        {showSimilarity && model.similarity !== undefined && (
          <div className="absolute top-1 right-1 bg-blue-600 text-white text-xs px-2 py-0.5 rounded-full">
            {(model.similarity * 100).toFixed(1)}%
          </div>
        )}
        
        {/* Indexed badge */}
        {model.isIndexed && (
          <div className="absolute top-1 left-1 bg-green-500 text-white text-xs px-2 py-0.5 rounded-full">
            Indexed
          </div>
        )}
      </div>

      {/* Info */}
      <div className={compact ? 'space-y-0.5' : 'space-y-1'}>
        <h3 className={`font-medium text-gray-800 truncate ${compact ? 'text-xs' : 'text-sm'}`}>
          {model.name || model.filename?.replace('.obj', '')}
        </h3>
        
        <p className={`text-gray-500 truncate ${compact ? 'text-xs' : 'text-xs'}`}>
          {model.category}
        </p>
        
        {!compact && model.numVertices && (
          <p className="text-xs text-gray-400">
            {model.numVertices?.toLocaleString()} vertices • {model.numFaces?.toLocaleString()} faces
          </p>
        )}
      </div>

      {/* Actions */}
      {onSearch && !compact && (
        <div className="mt-2 flex gap-1">
          <button
            onClick={handleSearch}
            className="flex-1 py-1 px-2 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
          >
            Find Similar
          </button>
        </div>
      )}
    </div>
  );
};

export default Model3DCard;
