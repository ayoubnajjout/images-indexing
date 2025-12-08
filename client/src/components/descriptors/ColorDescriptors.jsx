import React from 'react';
import { Card } from '../ui';

const ColorDescriptors = ({ descriptors }) => {
  if (!descriptors) return null;
  
  const { histogram, dominantColors } = descriptors;
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Color Descriptors</h3>
        <p className="text-sm text-slate-500 mt-1">Color distribution and dominant colors</p>
      </Card.Header>
      <Card.Body>
        {/* Histogram */}
        {histogram && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-slate-700 mb-3">Color Histogram</h4>
            <div className="flex items-end gap-2 h-32">
              {histogram.map((value, index) => (
                <div key={index} className="flex-1 flex flex-col items-center">
                  <div className="w-full bg-slate-100 rounded-t flex items-end" style={{ height: '100%' }}>
                    <div
                      className="w-full bg-indigo-500 rounded-t transition-all duration-500"
                      style={{ height: `${value * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-500 mt-1">{Math.round(value * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Dominant Colors */}
        {dominantColors && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-3">Dominant Colors</h4>
            <div className="space-y-3">
              {dominantColors.map((color, index) => (
                <div key={index} className="flex items-center gap-3">
                  <div
                    className="w-12 h-12 rounded-lg border border-slate-200 flex-shrink-0"
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
