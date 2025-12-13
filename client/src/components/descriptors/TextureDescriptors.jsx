import React from 'react';
import { Card } from '../ui';

const TextureDescriptors = ({ descriptors }) => {
  if (!descriptors) return null;
  
  const { tamura, gabor } = descriptors;

  // Gabor max for scaling
  const gaborMax = gabor ? Math.max(...gabor, 0.01) : 1;
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Texture Descriptors</h3>
      </Card.Header>
      <Card.Body className="space-y-6">
        {/* Tamura Features - Visual gauges */}
        {tamura && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-3">Tamura Features</h4>
            <div className="space-y-4">
              {/* Coarseness */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-slate-600">Coarseness</span>
                  <span className="text-sm font-medium text-slate-700">{(tamura.coarseness * 100).toFixed(0)}%</span>
                </div>
                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-400 to-blue-600 rounded-full transition-all"
                    style={{ width: `${tamura.coarseness * 100}%` }}
                  />
                </div>
                <div className="flex justify-between mt-1 text-xs text-slate-400">
                  <span>Fine</span>
                  <span>Coarse</span>
                </div>
              </div>
              
              {/* Contrast */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-slate-600">Contrast</span>
                  <span className="text-sm font-medium text-slate-700">{(tamura.contrast * 100).toFixed(0)}%</span>
                </div>
                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-400 to-purple-600 rounded-full transition-all"
                    style={{ width: `${tamura.contrast * 100}%` }}
                  />
                </div>
                <div className="flex justify-between mt-1 text-xs text-slate-400">
                  <span>Low</span>
                  <span>High</span>
                </div>
              </div>
              
              {/* Directionality */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-slate-600">Directionality</span>
                  <span className="text-sm font-medium text-slate-700">{(tamura.directionality * 100).toFixed(0)}%</span>
                </div>
                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-400 to-green-600 rounded-full transition-all"
                    style={{ width: `${tamura.directionality * 100}%` }}
                  />
                </div>
                <div className="flex justify-between mt-1 text-xs text-slate-400">
                  <span>Random</span>
                  <span>Directional</span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Gabor Features - Heatmap style */}
        {gabor && gabor.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-3">Gabor Filter Responses</h4>
            {/* Grid visualization - assuming 4 orientations x N scales */}
            <div className="grid grid-cols-4 gap-1 mb-2">
              {gabor.map((value, index) => {
                const intensity = value / gaborMax;
                return (
                  <div 
                    key={index} 
                    className="aspect-square rounded flex items-center justify-center text-xs font-mono transition-all hover:scale-110"
                    style={{ 
                      backgroundColor: `rgba(99, 102, 241, ${intensity})`,
                      color: intensity > 0.5 ? 'white' : '#64748b'
                    }}
                    title={`Filter ${index + 1}: ${value.toFixed(3)}`}
                  >
                    {value.toFixed(2)}
                  </div>
                );
              })}
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span>Response strength:</span>
              <div className="flex-1 h-2 rounded bg-gradient-to-r from-slate-100 to-indigo-500" />
              <span>Max: {gaborMax.toFixed(2)}</span>
            </div>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default TextureDescriptors;
