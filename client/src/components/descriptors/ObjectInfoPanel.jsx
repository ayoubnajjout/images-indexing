import React from 'react';
import { Card, Badge } from '../ui';

const ObjectInfoPanel = ({ image, object }) => {
  if (!image || !object) return null;
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Object Information</h3>
      </Card.Header>
      <Card.Body>
        {/* Image Thumbnail with Highlighted Object */}
        <div className="relative mb-4 rounded-lg overflow-hidden border border-slate-200">
          <img
            src={image.url}
            alt={image.filename}
            className="w-full h-auto"
          />
          {/* Simple overlay indicator */}
          <div className="absolute top-2 right-2">
            <Badge variant="primary">
              Selected Object
            </Badge>
          </div>
        </div>
        
        {/* Object Details */}
        <div className="space-y-3">
          <div>
            <span className="text-sm text-slate-500">Label</span>
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
            <span className="text-sm text-slate-500">Bounding Box</span>
            <p className="text-sm text-slate-700 mt-1 font-mono">
              [{object.bbox.map(v => Math.round(v)).join(', ')}]
            </p>
          </div>
          
          <div>
            <span className="text-sm text-slate-500">Image</span>
            <p className="text-sm text-slate-700 mt-1 truncate">
              {image.filename}
            </p>
          </div>
        </div>
      </Card.Body>
    </Card>
  );
};

export default ObjectInfoPanel;
