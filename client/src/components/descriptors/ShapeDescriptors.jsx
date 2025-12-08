import React from 'react';
import { Card, Badge } from '../ui';

const ShapeDescriptors = ({ descriptors }) => {
  if (!descriptors) return null;
  
  const { huMoments, orientationHistogram } = descriptors;
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Shape Descriptors</h3>
        <p className="text-sm text-slate-500 mt-1">Hu moments and orientation histogram</p>
      </Card.Header>
      <Card.Body>
        {/* Hu Moments */}
        {huMoments && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-slate-700 mb-3">Hu Moments</h4>
            <div className="grid grid-cols-2 gap-3">
              {huMoments.map((value, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-slate-50 rounded-lg">
                  <span className="text-sm text-slate-600">Moment {index + 1}</span>
                  <Badge variant="default">{value.toFixed(4)}</Badge>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Orientation Histogram */}
        {orientationHistogram && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-3">Orientation Histogram</h4>
            <div className="flex items-end gap-2 h-24">
              {orientationHistogram.map((value, index) => (
                <div key={index} className="flex-1 flex flex-col items-center">
                  <div className="w-full bg-slate-100 rounded-t flex items-end" style={{ height: '100%' }}>
                    <div
                      className="w-full bg-gradient-to-t from-orange-500 to-yellow-400 rounded-t transition-all duration-500"
                      style={{ height: `${value * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-500 mt-1">
                    {index * 30}°
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default ShapeDescriptors;
