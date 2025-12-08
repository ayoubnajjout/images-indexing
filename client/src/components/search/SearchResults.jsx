import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Badge } from '../ui';

const SearchResults = ({ results, loading }) => {
  const navigate = useNavigate();
  
  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {Array.from({ length: 6 }).map((_, index) => (
          <Card key={index} className="animate-pulse">
            <div className="aspect-video bg-slate-200" />
            <Card.Body>
              <div className="h-4 bg-slate-200 rounded mb-2" />
              <div className="h-3 bg-slate-200 rounded w-2/3" />
            </Card.Body>
          </Card>
        ))}
      </div>
    );
  }
  
  if (!results || results.length === 0) {
    return (
      <div className="text-center py-16">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-slate-100 rounded-full mb-4">
          <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-slate-900 mb-2">No results yet</h3>
        <p className="text-slate-500">Build a query and search to find similar objects</p>
      </div>
    );
  }
  
  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-900">
          Search Results ({results.length})
        </h3>
        <div className="text-sm text-slate-500">
          Sorted by relevance
        </div>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {results.map((result) => (
          <Card
            key={`${result.imageId}-${result.objectId}`}
            className="cursor-pointer"
            onClick={() => navigate(`/image/${result.imageId}`)}
          >
            {/* Image with overlay */}
            <div className="relative aspect-video overflow-hidden bg-slate-100">
              <img
                src={result.image.url}
                alt={result.image.filename}
                className="w-full h-full object-cover"
              />
              
              {/* Rank Badge */}
              <div className="absolute top-2 left-2">
                <Badge variant="primary">
                  #{result.rank}
                </Badge>
              </div>
              
              {/* Similarity Score */}
              <div className="absolute top-2 right-2">
                <Badge variant="success">
                  {Math.round(result.similarity * 100)}% match
                </Badge>
              </div>
              
              {/* Object Label */}
              {result.object && (
                <div className="absolute bottom-2 left-2">
                  <Badge variant="default" className="bg-black bg-opacity-70 text-white">
                    {result.object.label}
                  </Badge>
                </div>
              )}
            </div>
            
            {/* Info */}
            <Card.Body className="py-3">
              <h4 className="font-medium text-slate-900 truncate mb-1">
                {result.image.filename}
              </h4>
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <span>Similarity: {(result.similarity * 100).toFixed(1)}%</span>
                {result.object && (
                  <>
                    <span>•</span>
                    <span className="capitalize">{result.object.label}</span>
                  </>
                )}
              </div>
            </Card.Body>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default SearchResults;
