import React from 'react';
import { Card, Badge } from '../ui';

const QuerySummary = ({ image, object, descriptors }) => {
  if (!image || !object) return null;
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Query Summary</h3>
      </Card.Header>
      <Card.Body>
        <div className="flex gap-4">
          {/* Image Preview */}
          <div className="flex-shrink-0">
            <div className="w-32 h-32 rounded-lg overflow-hidden border border-slate-200">
              <img
                src={image.url}
                alt={image.filename}
                className="w-full h-full object-cover"
              />
            </div>
          </div>
          
          {/* Details */}
          <div className="flex-1 space-y-3">
            <div>
              <span className="text-sm text-slate-500">Object</span>
              <p className="text-base font-medium text-slate-900 capitalize mt-1">
                {object.label}
              </p>
            </div>
            
            <div>
              <span className="text-sm text-slate-500">Confidence</span>
              <div className="mt-1">
                <Badge variant="success">
                  {Math.round(object.confidence * 100)}%
                </Badge>
              </div>
            </div>
            
            <div>
              <span className="text-sm text-slate-500">Image</span>
              <p className="text-sm text-slate-700 mt-1 truncate">
                {image.filename}
              </p>
            </div>
            
            {/* Key Descriptors */}
            {descriptors?.color?.dominantColors && (
              <div>
                <span className="text-sm text-slate-500">Dominant Colors</span>
                <div className="flex gap-2 mt-1">
                  {descriptors.color.dominantColors.slice(0, 4).map((color, index) => (
                    <div
                      key={index}
                      className="w-8 h-8 rounded border border-slate-200"
                      style={{ backgroundColor: color.color }}
                      title={`${color.name} (${color.percentage}%)`}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </Card.Body>
    </Card>
  );
};

export default QuerySummary;
