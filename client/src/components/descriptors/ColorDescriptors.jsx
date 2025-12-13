import React from 'react';
import { Card } from '../ui';

const ColorDescriptors = ({ descriptors }) => {
  if (!descriptors) return null;
  
  const { histogram, hsHistogram, dominantColors, dominantColorsHS, labColorMoments } = descriptors;
  
  // Split histogram into H, S, V channels (16 bins each = 48 total)
  const binsPerChannel = histogram ? Math.floor(histogram.length / 3) : 0;
  const hHistogram = histogram?.slice(0, binsPerChannel) || [];
  const sHistogram = histogram?.slice(binsPerChannel, binsPerChannel * 2) || [];
  const vHistogram = histogram?.slice(binsPerChannel * 2) || [];

  // Get max values for proper scaling
  const hMax = Math.max(...hHistogram, 0.01);
  const sMax = Math.max(...sHistogram, 0.01);
  const vMax = Math.max(...vHistogram, 0.01);
  
  // Color gradients for each channel
  const getHueColor = (index, total) => {
    const hue = (index / total) * 360;
    return `hsl(${hue}, 80%, 50%)`;
  };
  
  const getSatColor = (index, total) => {
    const sat = 20 + (index / total) * 80;
    return `hsl(210, ${sat}%, 50%)`;
  };
  
  const getValColor = (index, total) => {
    const val = (index / total) * 100;
    return `hsl(0, 0%, ${val}%)`;
  };

  // LAB moment labels
  const labChannelNames = ['L', 'a', 'b'];
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Color Descriptors</h3>
      </Card.Header>
      <Card.Body className="space-y-6">
        {/* Dominant Colors - Main Visual */}
        {dominantColors && dominantColors.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-3">Dominant Colors</h4>
            {/* Color Bar */}
            <div className="h-12 rounded-lg overflow-hidden flex shadow-inner mb-3">
              {dominantColors.map((color, index) => (
                <div
                  key={index}
                  style={{ 
                    backgroundColor: color.color,
                    flex: color.percentage 
                  }}
                  className="transition-all hover:flex-grow-[1.2]"
                  title={`${color.name}: ${color.percentage}%`}
                />
              ))}
            </div>
            {/* Color Legend */}
            <div className="grid grid-cols-2 gap-2">
              {dominantColors.map((color, index) => (
                <div key={index} className="flex items-center gap-2">
                  <div
                    className="w-5 h-5 rounded shadow-sm border border-slate-200"
                    style={{ backgroundColor: color.color }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-slate-700 truncate">{color.name}</div>
                    <div className="text-xs text-slate-400">{color.percentage}%</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Hue Distribution */}
        {hHistogram.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-2">Hue Distribution</h4>
            <div className="relative">
              <div className="flex items-end gap-px h-24 bg-gradient-to-r from-slate-50 to-slate-100 rounded-lg p-2">
                {hHistogram.map((value, index) => (
                  <div 
                    key={index} 
                    className="flex-1 rounded-t transition-all duration-300 hover:opacity-80"
                    style={{ 
                      height: `${(value / hMax) * 100}%`,
                      backgroundColor: getHueColor(index, hHistogram.length),
                      minHeight: value > 0 ? '4px' : '1px'
                    }}
                    title={`${Math.round((index / hHistogram.length) * 360)}°: ${(value * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
              {/* Hue spectrum bar */}
              <div className="h-2 rounded-b-lg overflow-hidden flex mt-1">
                {Array.from({ length: hHistogram.length }).map((_, i) => (
                  <div 
                    key={i} 
                    className="flex-1" 
                    style={{ backgroundColor: getHueColor(i, hHistogram.length) }} 
                  />
                ))}
              </div>
            </div>
          </div>
        )}
        
        {/* Saturation & Brightness - Side by Side */}
        <div className="grid grid-cols-2 gap-4">
          {sHistogram.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-slate-700 mb-2">Saturation</h4>
              <div className="flex items-end gap-px h-16 bg-slate-50 rounded p-1">
                {sHistogram.map((value, index) => (
                  <div 
                    key={index} 
                    className="flex-1 rounded-t"
                    style={{ 
                      height: `${(value / sMax) * 100}%`,
                      backgroundColor: getSatColor(index, sHistogram.length),
                      minHeight: value > 0 ? '2px' : '1px'
                    }}
                    title={`${Math.round((index / sHistogram.length) * 100)}%: ${(value * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
              <div className="flex justify-between mt-1 text-xs text-slate-400">
                <span>Gray</span>
                <span>Vivid</span>
              </div>
            </div>
          )}
          
          {vHistogram.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-slate-700 mb-2">Brightness</h4>
              <div className="flex items-end gap-px h-16 bg-slate-50 rounded p-1">
                {vHistogram.map((value, index) => (
                  <div 
                    key={index} 
                    className="flex-1 rounded-t"
                    style={{ 
                      height: `${(value / vMax) * 100}%`,
                      backgroundColor: getValColor(index, vHistogram.length),
                      minHeight: value > 0 ? '2px' : '1px'
                    }}
                    title={`${Math.round((index / vHistogram.length) * 100)}%: ${(value * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
              <div className="flex justify-between mt-1 text-xs text-slate-400">
                <span>Dark</span>
                <span>Bright</span>
              </div>
            </div>
          )}
        </div>

        {/* LAB Color Moments - Compact */}
        {labColorMoments && labColorMoments.length === 9 && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-2">LAB Color Statistics</h4>
            <div className="grid grid-cols-3 gap-3">
              {labChannelNames.map((channel, channelIdx) => {
                const mean = labColorMoments[channelIdx * 3];
                const variance = labColorMoments[channelIdx * 3 + 1];
                const skewness = labColorMoments[channelIdx * 3 + 2];
                const bgColors = ['bg-gradient-to-br from-gray-50 to-gray-100', 'bg-gradient-to-br from-green-50 to-red-50', 'bg-gradient-to-br from-blue-50 to-yellow-50'];
                
                return (
                  <div key={channelIdx} className={`${bgColors[channelIdx]} rounded-lg p-3 text-center`}>
                    <div className="text-lg font-bold text-slate-700">{channel}</div>
                    <div className="text-xs text-slate-500 mt-1">
                      μ: {mean?.toFixed(1)}
                    </div>
                    <div className="text-xs text-slate-500">
                      σ²: {variance?.toFixed(1)}
                    </div>
                    <div className="text-xs text-slate-500">
                      γ: {skewness?.toFixed(2)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* HS Space Colors */}
        {dominantColorsHS && dominantColorsHS.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-2">HS Color Clusters</h4>
            <div className="flex flex-wrap gap-1">
              {dominantColorsHS.map((hs, index) => {
                const hue = hs[0] * 2; // H is 0-180 in OpenCV
                const sat = (hs[1] / 255) * 100;
                return (
                  <div
                    key={index}
                    className="w-8 h-8 rounded shadow-sm border border-slate-200"
                    style={{ backgroundColor: `hsl(${hue}, ${sat}%, 50%)` }}
                    title={`H: ${Math.round(hue)}°, S: ${Math.round(sat)}%`}
                  />
                );
              })}
            </div>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default ColorDescriptors;
