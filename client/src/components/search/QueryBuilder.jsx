import React from 'react';
import { Card, Badge, Button } from '../ui';

const QueryBuilder = ({ 
  images, 
  selectedImageId, 
  onImageSelect, 
  selectedObjectId, 
  onObjectSelect,
  onSearch,
  searching 
}) => {
  const selectedImage = images.find(img => img.id.toString() === selectedImageId);
  const objects = selectedImage?.detections || [];
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Build Your Query</h3>
        <p className="text-sm text-slate-500 mt-1">
          Select an image and object to find similar items
        </p>
      </Card.Header>
      <Card.Body>
        {/* Step 1: Select Image */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-slate-700 mb-3">
            Step 1: Select Query Image
          </h4>
          <div className="grid grid-cols-4 gap-3">
            {images.slice(0, 8).map((image) => (
              <button
                key={image.id}
                onClick={() => onImageSelect(image.id.toString())}
                className={`relative aspect-video rounded-lg overflow-hidden border-2 transition ${
                  selectedImageId === image.id.toString()
                    ? 'border-indigo-500 ring-2 ring-indigo-200'
                    : 'border-slate-200 hover:border-slate-300'
                }`}
              >
                <img
                  src={image.url}
                  alt={image.filename}
                  className="w-full h-full object-cover"
                />
                {selectedImageId === image.id.toString() && (
                  <div className="absolute top-1 right-1">
                    <Badge variant="primary" size="sm">✓</Badge>
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
        
        {/* Step 2: Select Object */}
        {selectedImageId && objects.length > 0 && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-slate-700 mb-3">
              Step 2: Select Object from Image
            </h4>
            <div className="space-y-2">
              {objects.map((object) => (
                <button
                  key={object.id}
                  onClick={() => onObjectSelect(object.id.toString())}
                  className={`w-full p-3 rounded-lg border-2 text-left transition ${
                    selectedObjectId === object.id.toString()
                      ? 'border-indigo-500 bg-indigo-50'
                      : 'border-slate-200 hover:border-slate-300 bg-white'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="font-medium text-slate-900 capitalize">
                        {object.label}
                      </span>
                      <span className="text-sm text-slate-500 ml-2">
                        ({Math.round(object.confidence * 100)}% confidence)
                      </span>
                    </div>
                    {selectedObjectId === object.id.toString() && (
                      <Badge variant="primary">Selected</Badge>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
        
        {/* Step 3: Search Button */}
        {selectedImageId && selectedObjectId && (
          <div>
            <Button
              variant="primary"
              size="lg"
              onClick={onSearch}
              disabled={searching}
              className="w-full"
            >
              {searching ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Searching...
                </>
              ) : (
                <>
                  <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  Search Similar Objects
                </>
              )}
            </Button>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default QueryBuilder;
