import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Badge, Button } from '../ui';

const ObjectList = ({ detections, selectedObjectId, onObjectSelect, imageId }) => {
  const navigate = useNavigate();
  
  const handleViewDescriptors = (objectId) => {
    navigate(`/descriptors?imageId=${imageId}&objectId=${objectId}`);
  };
  
  const handleUseAsQuery = (objectId) => {
    navigate(`/search?imageId=${imageId}&objectId=${objectId}`);
  };

  // Helper to get display label (prefer readable name)
  const getDisplayLabel = (detection) => {
    if (detection.label_readable) return detection.label_readable;
    if (detection.label) return detection.label;
    return 'Unknown';
  };
  
  if (!detections || detections.length === 0) {
    return (
      <Card>
        <Card.Body>
          <div className="text-center py-8">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-slate-100 rounded-full mb-3">
              <svg className="w-6 h-6 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-slate-600">No objects detected yet</p>
          </div>
        </Card.Body>
      </Card>
    );
  }
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">
          Detected Objects ({detections.length})
        </h3>
      </Card.Header>
      <Card.Body className="p-0">
        <div className="divide-y divide-slate-200">
          {detections.map((detection) => (
            <div
              key={detection.id}
              className={`p-4 transition cursor-pointer ${
                selectedObjectId === detection.id
                  ? 'bg-indigo-50'
                  : 'hover:bg-slate-50'
              }`}
              onClick={() => onObjectSelect(detection.id)}
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h4 className="font-medium text-slate-900 capitalize">
                    {getDisplayLabel(detection)}
                  </h4>
                  {detection.label && detection.label !== detection.label_readable && (
                    <span className="text-xs text-slate-400 font-mono">{detection.label}</span>
                  )}
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant="success" size="sm">
                      {Math.round(detection.confidence * 100)}% confident
                    </Badge>
                    {detection.class_id !== undefined && (
                      <Badge variant="default" size="sm">
                        Class {detection.class_id}
                      </Badge>
                    )}
                    {selectedObjectId === detection.id && (
                      <Badge variant="primary" size="sm">
                        Selected
                      </Badge>
                    )}
                  </div>
                </div>
                
                {/* Color indicator */}
                <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-purple-500 rounded" />
              </div>
              
              {/* Bounding box info */}
              <div className="text-xs text-slate-500 mb-3">
                BBox: [{detection.bbox.map(v => Math.round(v)).join(', ')}]
              </div>
              
              {/* Actions */}
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleViewDescriptors(detection.id);
                  }}
                >
                  View Descriptors
                </Button>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleUseAsQuery(detection.id);
                  }}
                >
                  Use as Query
                </Button>
              </div>
            </div>
          ))}
        </div>
      </Card.Body>
    </Card>
  );
};

export default ObjectList;
