/**
 * Descriptor Visualization Component
 * Displays shape descriptors as histograms and charts
 */

import { useState, useEffect } from 'react';

/**
 * Simple bar chart for histogram visualization
 */
const HistogramChart = ({ data, title, color = '#3b82f6', maxBars = 64 }) => {
  if (!data || data.length === 0) {
    return (
      <div className="text-center text-gray-400 text-sm py-4">
        No data available
      </div>
    );
  }

  const maxVal = Math.max(...data, 0.001);
  const displayData = data.length > maxBars ? 
    data.filter((_, i) => i % Math.ceil(data.length / maxBars) === 0) : data;
  
  return (
    <div className="w-full">
      <div className="text-xs text-gray-600 mb-1">{title}</div>
      <div className="flex items-end h-16 gap-px bg-gray-100 rounded p-1">
        {displayData.map((val, i) => (
          <div
            key={i}
            className="flex-1 rounded-t transition-all hover:opacity-80"
            style={{
              height: `${Math.max((val / maxVal) * 100, 2)}%`,
              backgroundColor: color,
              minWidth: '2px'
            }}
            title={`Bin ${i}: ${val.toFixed(4)}`}
          />
        ))}
      </div>
      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>0</span>
        <span>{data.length} bins</span>
      </div>
    </div>
  );
};

/**
 * Vector/Feature visualization as horizontal bars
 */
const VectorChart = ({ data, labels, title, color = '#10b981' }) => {
  if (!data || data.length === 0) {
    return (
      <div className="text-center text-gray-400 text-sm py-4">
        No data available
      </div>
    );
  }

  const maxVal = Math.max(...data.map(Math.abs), 0.001);
  
  return (
    <div className="w-full">
      <div className="text-xs text-gray-600 mb-2">{title}</div>
      <div className="space-y-1">
        {data.map((val, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="w-20 text-xs text-gray-500 truncate">
              {labels?.[i] || `Feature ${i + 1}`}
            </div>
            <div className="flex-1 h-4 bg-gray-100 rounded overflow-hidden">
              <div
                className="h-full rounded transition-all"
                style={{
                  width: `${Math.abs(val / maxVal) * 100}%`,
                  backgroundColor: val >= 0 ? color : '#ef4444'
                }}
              />
            </div>
            <div className="w-16 text-xs text-gray-600 text-right">
              {val.toFixed(3)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Similarity comparison bar
 */
const SimilarityBar = ({ name, score, weight }) => {
  const percentage = (score * 100).toFixed(1);
  const getColor = (s) => {
    if (s >= 0.8) return '#22c55e'; // green
    if (s >= 0.6) return '#eab308'; // yellow
    if (s >= 0.4) return '#f97316'; // orange
    return '#ef4444'; // red
  };
  
  return (
    <div className="flex items-center gap-2 py-1">
      <div className="w-24 text-xs text-gray-600 font-medium">{name}</div>
      <div className="flex-1 h-5 bg-gray-100 rounded overflow-hidden relative">
        <div
          className="h-full rounded transition-all"
          style={{
            width: `${score * 100}%`,
            backgroundColor: getColor(score)
          }}
        />
        <span className="absolute inset-0 flex items-center justify-center text-xs font-medium text-gray-700">
          {percentage}%
        </span>
      </div>
      <div className="w-12 text-xs text-gray-400 text-right">
        w: {(weight * 100).toFixed(0)}%
      </div>
    </div>
  );
};

/**
 * Main Descriptor Visualization Component
 */
const DescriptorVisualization = ({ 
  descriptors, 
  modelInfo,
  comparison,
  compact = false 
}) => {
  const [activeTab, setActiveTab] = useState('histograms');
  
  if (!descriptors) {
    return (
      <div className="text-center text-gray-500 py-8">
        Select an indexed model to view its descriptors
      </div>
    );
  }

  const histogramDescriptors = ['d1', 'd2', 'd3', 'd4', 'a3'];
  const vectorDescriptors = ['bbox', 'moments', 'mesh_stats'];
  
  const colors = {
    d1: '#3b82f6', // blue
    d2: '#8b5cf6', // purple
    d3: '#ec4899', // pink
    d4: '#f97316', // orange
    a3: '#10b981', // green
    bbox: '#06b6d4', // cyan
    moments: '#6366f1', // indigo
    mesh_stats: '#84cc16' // lime
  };

  return (
    <div className={`bg-white rounded-lg shadow ${compact ? 'p-3' : 'p-4'}`}>
      {/* Model Info Header */}
      {modelInfo && (
        <div className="mb-4 pb-3 border-b">
          <h3 className="font-medium text-gray-800">{modelInfo.name}</h3>
          <p className="text-sm text-gray-500">{modelInfo.category}</p>
          <div className="flex gap-4 mt-2 text-xs text-gray-500">
            {modelInfo.numVertices && (
              <span>{modelInfo.numVertices.toLocaleString()} vertices</span>
            )}
            {modelInfo.numFaces && (
              <span>{modelInfo.numFaces.toLocaleString()} faces</span>
            )}
            {modelInfo.surfaceArea && (
              <span>Area: {modelInfo.surfaceArea.toFixed(4)}</span>
            )}
          </div>
        </div>
      )}

      {/* Comparison Results */}
      {comparison && (
        <div className="mb-4 pb-3 border-b">
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            Similarity Breakdown
          </h4>
          <div className="space-y-1">
            {Object.entries(comparison).map(([key, val]) => (
              <SimilarityBar
                key={key}
                name={key.toUpperCase()}
                score={val.score}
                weight={val.weight}
              />
            ))}
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 mb-3 border-b">
        <button
          onClick={() => setActiveTab('histograms')}
          className={`px-3 py-1.5 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'histograms'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Shape Distributions
        </button>
        <button
          onClick={() => setActiveTab('vectors')}
          className={`px-3 py-1.5 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'vectors'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Geometric Features
        </button>
      </div>

      {/* Content */}
      {activeTab === 'histograms' && (
        <div className={`grid ${compact ? 'grid-cols-1 gap-3' : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'}`}>
          {histogramDescriptors.map((key) => {
            const desc = descriptors[key];
            if (!desc) return null;
            return (
              <div key={key} className="bg-gray-50 rounded p-3">
                <HistogramChart
                  data={desc.data}
                  title={desc.name}
                  color={colors[key]}
                />
                <p className="text-xs text-gray-400 mt-2">{desc.description}</p>
              </div>
            );
          })}
        </div>
      )}

      {activeTab === 'vectors' && (
        <div className={`grid ${compact ? 'grid-cols-1 gap-3' : 'grid-cols-1 md:grid-cols-2 gap-4'}`}>
          {vectorDescriptors.map((key) => {
            const desc = descriptors[key];
            if (!desc || !desc.data || desc.data.length === 0) return null;
            return (
              <div key={key} className="bg-gray-50 rounded p-3">
                <VectorChart
                  data={desc.data}
                  labels={desc.labels}
                  title={desc.name}
                  color={colors[key]}
                />
                <p className="text-xs text-gray-400 mt-2">{desc.description}</p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default DescriptorVisualization;
