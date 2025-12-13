import React from 'react';
import { Card, Badge } from '../ui';

const ShapeDescriptors = ({ descriptors }) => {
  if (!descriptors) return null;
  
  const { 
    huMoments, 
    contourHuMoments,
    orientationHistogram, 
    orientationHistogramContour,
    shapeMetrics,
    fourierDescriptors 
  } = descriptors;

  // Use contour-based if available, otherwise fall back to image-based
  const displayHuMoments = contourHuMoments || huMoments;
  const displayOrientationHist = orientationHistogramContour || orientationHistogram;
  const orientationBins = displayOrientationHist?.length || 8;
  const degreesPerBin = 360 / orientationBins;

  // Get max for proper scaling
  const orientMax = displayOrientationHist ? Math.max(...displayOrientationHist, 0.01) : 1;
  const fourierMax = fourierDescriptors ? Math.max(...fourierDescriptors.slice(0, 20), 0.01) : 1;
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Shape Descriptors</h3>
      </Card.Header>
      <Card.Body className="space-y-6">
        {/* Shape Metrics - Visual Cards */}
        {shapeMetrics && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-3">Shape Metrics</h4>
            <div className="grid grid-cols-3 gap-3">
              {/* Compactness - Circle visualization */}
              <div className="bg-gradient-to-br from-indigo-50 to-blue-50 rounded-lg p-4 text-center">
                <div className="w-16 h-16 mx-auto relative mb-2">
                  <div className="absolute inset-0 rounded-full border-4 border-indigo-200" />
                  <div 
                    className="absolute rounded-full bg-indigo-500"
                    style={{ 
                      width: `${shapeMetrics.compactness * 100}%`,
                      height: `${shapeMetrics.compactness * 100}%`,
                      top: `${(1 - shapeMetrics.compactness) * 50}%`,
                      left: `${(1 - shapeMetrics.compactness) * 50}%`
                    }}
                  />
                </div>
                <div className="text-2xl font-bold text-indigo-600">
                  {(shapeMetrics.compactness * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-slate-600">Compactness</div>
                <div className="text-xs text-slate-400">Circle = 100%</div>
              </div>
              
              {/* Aspect Ratio - Rectangle visualization */}
              <div className="bg-gradient-to-br from-emerald-50 to-green-50 rounded-lg p-4 text-center">
                <div className="w-16 h-16 mx-auto flex items-center justify-center mb-2">
                  <div 
                    className="bg-emerald-500 rounded"
                    style={{ 
                      width: shapeMetrics.aspectRatio >= 1 ? '100%' : `${shapeMetrics.aspectRatio * 100}%`,
                      height: shapeMetrics.aspectRatio >= 1 ? `${(1/shapeMetrics.aspectRatio) * 100}%` : '100%',
                      maxWidth: '56px',
                      maxHeight: '56px'
                    }}
                  />
                </div>
                <div className="text-2xl font-bold text-emerald-600">
                  {shapeMetrics.aspectRatio?.toFixed(2)}
                </div>
                <div className="text-xs text-slate-600">Aspect Ratio</div>
                <div className="text-xs text-slate-400">W / H</div>
              </div>
              
              {/* Solidity - Fill visualization */}
              <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-4 text-center">
                <div className="w-16 h-16 mx-auto relative mb-2 rounded bg-amber-100 overflow-hidden">
                  <div 
                    className="absolute bottom-0 left-0 right-0 bg-amber-500 transition-all"
                    style={{ height: `${shapeMetrics.solidity * 100}%` }}
                  />
                </div>
                <div className="text-2xl font-bold text-amber-600">
                  {(shapeMetrics.solidity * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-slate-600">Solidity</div>
                <div className="text-xs text-slate-400">Area / Hull</div>
              </div>
            </div>
          </div>
        )}

        {/* Orientation Histogram - Radial */}
        {displayOrientationHist && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-3">Edge Orientations</h4>
            <div className="flex gap-4 items-center">
              {/* Radial visualization */}
              <div className="w-32 h-32 relative">
                <svg viewBox="0 0 100 100" className="w-full h-full">
                  {/* Background circle */}
                  <circle cx="50" cy="50" r="45" fill="none" stroke="#e2e8f0" strokeWidth="1" />
                  <circle cx="50" cy="50" r="30" fill="none" stroke="#e2e8f0" strokeWidth="1" />
                  <circle cx="50" cy="50" r="15" fill="none" stroke="#e2e8f0" strokeWidth="1" />
                  
                  {/* Direction lines */}
                  {displayOrientationHist.map((value, index) => {
                    const angle = (index * degreesPerBin - 90) * (Math.PI / 180);
                    const length = (value / orientMax) * 40 + 5;
                    const x2 = 50 + Math.cos(angle) * length;
                    const y2 = 50 + Math.sin(angle) * length;
                    return (
                      <line
                        key={index}
                        x1="50"
                        y1="50"
                        x2={x2}
                        y2={y2}
                        stroke={`hsl(${30 + (value / orientMax) * 30}, 80%, 50%)`}
                        strokeWidth="4"
                        strokeLinecap="round"
                        opacity="0.9"
                      />
                    );
                  })}
                  
                  {/* Center dot */}
                  <circle cx="50" cy="50" r="3" fill="#64748b" />
                </svg>
              </div>
              
              {/* Bar chart */}
              <div className="flex-1">
                <div className="flex items-end gap-1 h-20">
                  {displayOrientationHist.map((value, index) => (
                    <div key={index} className="flex-1 flex flex-col items-center">
                      <div
                        className="w-full bg-gradient-to-t from-orange-500 to-amber-400 rounded-t"
                        style={{ height: `${(value / orientMax) * 100}%`, minHeight: '2px' }}
                        title={`${Math.round(index * degreesPerBin)}°: ${(value * 100).toFixed(1)}%`}
                      />
                    </div>
                  ))}
                </div>
                <div className="flex justify-between mt-1 text-xs text-slate-400">
                  <span>0°</span>
                  <span>180°</span>
                  <span>360°</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Hu Moments - Compact Grid */}
        {displayHuMoments && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-2">
              Hu Moments <span className="text-xs text-slate-400 font-normal">(rotation invariant)</span>
            </h4>
            <div className="flex gap-2 flex-wrap">
              {displayHuMoments.map((value, index) => (
                <div key={index} className="bg-slate-50 rounded px-3 py-1.5 text-center">
                  <div className="text-xs text-slate-400">φ{index + 1}</div>
                  <div className="text-sm font-mono text-slate-700">{value?.toExponential(2)}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Fourier Descriptors - Frequency visualization */}
        {fourierDescriptors && fourierDescriptors.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-2">
              Fourier Descriptors <span className="text-xs text-slate-400 font-normal">(contour shape)</span>
            </h4>
            <div className="flex items-end gap-px h-16 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-2">
              {fourierDescriptors.slice(0, 20).map((value, index) => (
                <div 
                  key={index} 
                  className="flex-1 rounded-t bg-gradient-to-t from-purple-600 to-pink-400"
                  style={{ 
                    height: `${(value / fourierMax) * 100}%`,
                    minHeight: value > 0 ? '2px' : '1px',
                    opacity: 0.7 + (value / fourierMax) * 0.3
                  }}
                  title={`F${index + 1}: ${value?.toFixed(4)}`}
                />
              ))}
            </div>
            <div className="flex justify-between mt-1 text-xs text-slate-400">
              <span>Low freq (overall)</span>
              <span>High freq (detail)</span>
            </div>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default ShapeDescriptors;
