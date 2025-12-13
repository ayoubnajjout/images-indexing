import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Badge } from '../ui';

const SearchResults = ({ results, loading, queryObject }) => {
  const navigate = useNavigate();
  const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'detailed'
  
  if (loading) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {Array.from({ length: 8 }).map((_, index) => (
          <Card key={index} className="animate-pulse">
            <div className="aspect-square bg-slate-200" />
            <Card.Body className="p-3">
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

  // Helper to get display label
  const getDisplayLabel = (obj) => {
    if (obj?.label_readable) return obj.label_readable;
    if (obj?.label) return obj.label;
    return 'Unknown';
  };

  // Get dominant colors
  const getDominantColors = (object) => {
    return object?.descriptors?.color?.dominantColors || [];
  };

  // Get shape metrics
  const getShapeMetrics = (object) => {
    return object?.descriptors?.shape?.shapeMetrics || null;
  };

  // Get texture features
  const getTamura = (object) => {
    return object?.descriptors?.texture?.tamura || null;
  };

  // Render color bar
  const ColorBar = ({ colors, height = 'h-6' }) => {
    if (!colors || colors.length === 0) return <div className={`${height} bg-slate-100 rounded`} />;
    return (
      <div className={`flex ${height} rounded overflow-hidden shadow-inner`}>
        {colors.slice(0, 5).map((color, idx) => (
          <div
            key={idx}
            style={{ backgroundColor: color.color, flex: color.percentage }}
            title={`${color.name}: ${color.percentage}%`}
          />
        ))}
      </div>
    );
  };

  // Render similarity gauge
  const SimilarityGauge = ({ value, color, label }) => (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-500 w-12">{label}</span>
      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
        <div 
          className={`h-full rounded-full ${color}`}
          style={{ width: `${(value || 0) * 100}%` }}
        />
      </div>
      <span className="text-xs font-medium text-slate-700 w-10 text-right">
        {((value || 0) * 100).toFixed(0)}%
      </span>
    </div>
  );
  
  return (
    <div>
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-900">
          Results ({results.length})
        </h3>
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('grid')}
            className={`px-3 py-1.5 text-sm rounded-lg transition ${viewMode === 'grid' ? 'bg-indigo-100 text-indigo-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
          >
            Grid
          </button>
          <button
            onClick={() => setViewMode('detailed')}
            className={`px-3 py-1.5 text-sm rounded-lg transition ${viewMode === 'detailed' ? 'bg-indigo-100 text-indigo-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
          >
            Detailed
          </button>
        </div>
      </div>

      {/* Grid View */}
      {viewMode === 'grid' && (
        <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {results.map((result) => {
            const dominantColors = getDominantColors(result.object);
            const shapeMetrics = getShapeMetrics(result.object);
            
            return (
              <Card
                key={`${result.imageId}-${result.objectId}`}
                className="cursor-pointer hover:shadow-lg transition-all hover:-translate-y-1"
                onClick={() => navigate(`/image/${result.imageId}`)}
              >
                {/* Image */}
                <div className="relative aspect-square overflow-hidden bg-slate-100">
                  <img
                    src={result.image.url}
                    alt={result.image.filename}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-2 left-2 right-2 flex justify-between">
                    <Badge variant="default" className="bg-black/70 text-white text-xs">
                      #{result.rank}
                    </Badge>
                    <Badge variant="success" className="text-xs font-bold">
                      {Math.round(result.similarity * 100)}%
                    </Badge>
                  </div>
                  {result.object && (
                    <div className="absolute bottom-2 left-2">
                      <Badge variant="primary" className="text-xs">
                        {getDisplayLabel(result.object)}
                      </Badge>
                    </div>
                  )}
                </div>
                
                {/* Quick Info */}
                <Card.Body className="p-2 space-y-1.5">
                  <ColorBar colors={dominantColors} height="h-4" />
                  {result.scores && (
                    <div className="flex gap-1 text-xs">
                      <div className="flex-1 bg-blue-50 rounded px-1.5 py-0.5 text-center" title="Color">
                        <span className="text-blue-600 font-medium">{((result.scores.color || 0) * 100).toFixed(0)}%</span>
                      </div>
                      <div className="flex-1 bg-green-50 rounded px-1.5 py-0.5 text-center" title="Texture">
                        <span className="text-green-600 font-medium">{((result.scores.texture || 0) * 100).toFixed(0)}%</span>
                      </div>
                      <div className="flex-1 bg-orange-50 rounded px-1.5 py-0.5 text-center" title="Shape">
                        <span className="text-orange-600 font-medium">{((result.scores.shape || 0) * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  )}
                </Card.Body>
              </Card>
            );
          })}
        </div>
      )}

      {/* Detailed View */}
      {viewMode === 'detailed' && (
        <div className="space-y-4">
          {results.map((result) => {
            const dominantColors = getDominantColors(result.object);
            const shapeMetrics = getShapeMetrics(result.object);
            const tamura = getTamura(result.object);
            
            return (
              <Card
                key={`${result.imageId}-${result.objectId}`}
                className="cursor-pointer hover:shadow-lg transition-all"
                onClick={() => navigate(`/image/${result.imageId}`)}
              >
                <div className="flex">
                  {/* Image */}
                  <div className="relative w-48 h-48 flex-shrink-0">
                    <img
                      src={result.image.url}
                      alt={result.image.filename}
                      className="w-full h-full object-cover rounded-l-lg"
                    />
                    <div className="absolute top-2 left-2">
                      <Badge variant="default" className="bg-black/70 text-white">
                        #{result.rank}
                      </Badge>
                    </div>
                  </div>
                  
                  {/* Details */}
                  <Card.Body className="flex-1 p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h4 className="font-semibold text-slate-900">
                          {getDisplayLabel(result.object)}
                        </h4>
                        <p className="text-sm text-slate-500">{result.image.filename}</p>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-indigo-600">
                          {Math.round(result.similarity * 100)}%
                        </div>
                        <div className="text-xs text-slate-500">match</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      {/* Left: Similarity Scores */}
                      <div className="space-y-2">
                        <h5 className="text-xs font-medium text-slate-700 uppercase tracking-wide">Similarity</h5>
                        {result.scores && (
                          <>
                            <SimilarityGauge value={result.scores.color} color="bg-blue-500" label="Color" />
                            <SimilarityGauge value={result.scores.texture} color="bg-green-500" label="Texture" />
                            <SimilarityGauge value={result.scores.shape} color="bg-orange-500" label="Shape" />
                          </>
                        )}
                      </div>

                      {/* Right: Visual Descriptors */}
                      <div className="space-y-3">
                        {/* Dominant Colors */}
                        <div>
                          <h5 className="text-xs font-medium text-slate-700 uppercase tracking-wide mb-1">Colors</h5>
                          <ColorBar colors={dominantColors} height="h-8" />
                          {dominantColors.length > 0 && (
                            <div className="flex gap-1 mt-1 flex-wrap">
                              {dominantColors.slice(0, 4).map((c, i) => (
                                <span key={i} className="text-xs text-slate-500">{c.name} ({c.percentage}%)</span>
                              ))}
                            </div>
                          )}
                        </div>

                        {/* Shape Metrics */}
                        {shapeMetrics && (
                          <div>
                            <h5 className="text-xs font-medium text-slate-700 uppercase tracking-wide mb-1">Shape</h5>
                            <div className="flex gap-3 text-xs">
                              <div className="text-center">
                                <div className="w-8 h-8 mx-auto rounded-full border-2 border-indigo-300 flex items-center justify-center">
                                  <div 
                                    className="rounded-full bg-indigo-500"
                                    style={{ 
                                      width: `${shapeMetrics.compactness * 100}%`,
                                      height: `${shapeMetrics.compactness * 100}%`
                                    }}
                                  />
                                </div>
                                <div className="mt-0.5 text-slate-600">{(shapeMetrics.compactness * 100).toFixed(0)}%</div>
                                <div className="text-slate-400">Compact</div>
                              </div>
                              <div className="text-center">
                                <div className="w-8 h-8 mx-auto flex items-center justify-center">
                                  <div 
                                    className="bg-emerald-500 rounded"
                                    style={{ 
                                      width: shapeMetrics.aspectRatio >= 1 ? '100%' : `${shapeMetrics.aspectRatio * 100}%`,
                                      height: shapeMetrics.aspectRatio >= 1 ? `${(1/shapeMetrics.aspectRatio) * 100}%` : '100%'
                                    }}
                                  />
                                </div>
                                <div className="mt-0.5 text-slate-600">{shapeMetrics.aspectRatio?.toFixed(2)}</div>
                                <div className="text-slate-400">Ratio</div>
                              </div>
                              <div className="text-center">
                                <div className="w-8 h-8 mx-auto bg-amber-100 rounded overflow-hidden">
                                  <div 
                                    className="bg-amber-500 w-full"
                                    style={{ height: `${shapeMetrics.solidity * 100}%`, marginTop: `${(1 - shapeMetrics.solidity) * 100}%` }}
                                  />
                                </div>
                                <div className="mt-0.5 text-slate-600">{(shapeMetrics.solidity * 100).toFixed(0)}%</div>
                                <div className="text-slate-400">Solid</div>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Texture */}
                        {tamura && (
                          <div>
                            <h5 className="text-xs font-medium text-slate-700 uppercase tracking-wide mb-1">Texture</h5>
                            <div className="flex gap-2 text-xs">
                              <div className="flex-1 bg-slate-50 rounded p-1.5 text-center">
                                <div className="font-medium text-slate-700">{(tamura.coarseness * 100).toFixed(0)}%</div>
                                <div className="text-slate-400">Coarse</div>
                              </div>
                              <div className="flex-1 bg-slate-50 rounded p-1.5 text-center">
                                <div className="font-medium text-slate-700">{(tamura.contrast * 100).toFixed(0)}%</div>
                                <div className="text-slate-400">Contrast</div>
                              </div>
                              <div className="flex-1 bg-slate-50 rounded p-1.5 text-center">
                                <div className="font-medium text-slate-700">{(tamura.directionality * 100).toFixed(0)}%</div>
                                <div className="text-slate-400">Direction</div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </Card.Body>
                </div>
              </Card>
            );
          })}
        </div>
      )}
      
      {/* Legend */}
      <div className="mt-6 p-4 bg-slate-50 rounded-lg">
        <h4 className="text-sm font-medium text-slate-700 mb-2">Legend</h4>
        <div className="flex flex-wrap gap-6 text-xs text-slate-600">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span><strong>Color</strong> - Histogram & dominant colors similarity</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span><strong>Texture</strong> - Tamura & Gabor features similarity</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-orange-500" />
            <span><strong>Shape</strong> - Hu moments & contour similarity</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchResults;
