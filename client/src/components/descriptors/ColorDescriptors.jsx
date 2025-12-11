import React from 'react';
import { Card } from '../ui';

const ColorDescriptors = ({ descriptors }) => {
  if (!descriptors) return null;
  
  const { histogram, dominantColors } = descriptors;
  
  // Split histogram into H, S, V channels (assuming 16 bins each = 48 total)
  const binsPerChannel = histogram ? Math.floor(histogram.length / 3) : 0;
  const hHistogram = histogram?.slice(0, binsPerChannel) || [];
  const sHistogram = histogram?.slice(binsPerChannel, binsPerChannel * 2) || [];
  const vHistogram = histogram?.slice(binsPerChannel * 2) || [];
  
  // Color gradients for each channel
  const getHueColor = (index, total) => {
    const hue = (index / total) * 360;
    return `hsl(${hue}, 70%, 50%)`;
  };
  
  const getSatColor = (index, total) => {
    const sat = (index / total) * 100;
    return `hsl(200, ${sat}%, 50%)`;
  };
  
  const getValColor = (index, total) => {
    const val = (index / total) * 100;
    return `hsl(0, 0%, ${val}%)`;
  };
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Color Descriptors</h3>
        <p className="text-sm text-slate-500 mt-1">HSV color distribution and dominant colors</p>
      </Card.Header>
      <Card.Body>
        {/* Hue Histogram */}
        {hHistogram.length > 0 && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-slate-700 mb-2">Hue Distribution</h4>
            <div className="flex items-end gap-0.5 h-20 bg-slate-50 rounded p-2">
              {hHistogram.map((value, index) => (
                <div 
                  key={index} 
                  className="flex-1 rounded-t transition-all duration-300"
                  style={{ 
                    height: `${Math.max(value * 100, 2)}%`,
                    backgroundColor: getHueColor(index, hHistogram.length),
                    minHeight: '2px'
                  }}
                  title={`Hue ${index}: ${(value * 100).toFixed(1)}%`}
                />
              ))}
            </div>
            <div className="flex justify-between mt-1">
              <span className="text-xs text-slate-400">Red</span>
              <span className="text-xs text-slate-400">Yellow</span>
              <span className="text-xs text-slate-400">Green</span>
              <span className="text-xs text-slate-400">Cyan</span>
              <span className="text-xs text-slate-400">Blue</span>
              <span className="text-xs text-slate-400">Magenta</span>
            </div>
          </div>
        )}
        
        {/* Saturation Histogram */}
        {sHistogram.length > 0 && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-slate-700 mb-2">Saturation Distribution</h4>
            <div className="flex items-end gap-0.5 h-16 bg-slate-50 rounded p-2">
              {sHistogram.map((value, index) => (
                <div 
                  key={index} 
                  className="flex-1 rounded-t transition-all duration-300"
                  style={{ 
                    height: `${Math.max(value * 100, 2)}%`,
                    backgroundColor: getSatColor(index, sHistogram.length),
                    minHeight: '2px'
                  }}
                  title={`Saturation ${index}: ${(value * 100).toFixed(1)}%`}
                />
              ))}
            </div>
            <div className="flex justify-between mt-1">
              <span className="text-xs text-slate-400">Gray</span>
              <span className="text-xs text-slate-400">Vivid</span>
            </div>
          </div>
        )}
        
        {/* Value/Brightness Histogram */}
        {vHistogram.length > 0 && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-slate-700 mb-2">Brightness Distribution</h4>
            <div className="flex items-end gap-0.5 h-16 bg-slate-50 rounded p-2">
              {vHistogram.map((value, index) => (
                <div 
                  key={index} 
                  className="flex-1 rounded-t transition-all duration-300"
                  style={{ 
                    height: `${Math.max(value * 100, 2)}%`,
                    backgroundColor: getValColor(index, vHistogram.length),
                    minHeight: '2px'
                  }}
                  title={`Brightness ${index}: ${(value * 100).toFixed(1)}%`}
                />
              ))}
            </div>
            <div className="flex justify-between mt-1">
              <span className="text-xs text-slate-400">Dark</span>
              <span className="text-xs text-slate-400">Bright</span>
            </div>
          </div>
        )}
        
        {/* Dominant Colors */}
        {dominantColors && dominantColors.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-3">Dominant Colors</h4>
            <div className="space-y-3">
              {dominantColors.map((color, index) => (
                <div key={index} className="flex items-center gap-3">
                  <div
                    className="w-12 h-12 rounded-lg border border-slate-200 flex-shrink-0 shadow-sm"
                    style={{ backgroundColor: color.color }}
                  />
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-slate-900">{color.name}</span>
                      <span className="text-sm text-slate-600">{color.percentage}%</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2">
                      <div
                        className="h-2 rounded-full transition-all duration-500"
                        style={{
                          width: `${color.percentage}%`,
                          backgroundColor: color.color,
                        }}
                      />
                    </div>
                    <span className="text-xs text-slate-400 mt-0.5">{color.color}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default ColorDescriptors;
