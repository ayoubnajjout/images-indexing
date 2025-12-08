import React from 'react';
import { Card, Badge } from '../ui';

const TextureDescriptors = ({ descriptors }) => {
  if (!descriptors) return null;
  
  const { tamura, gabor } = descriptors;
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Texture Descriptors</h3>
        <p className="text-sm text-slate-500 mt-1">Tamura features and Gabor filters</p>
      </Card.Header>
      <Card.Body>
        {/* Tamura Features */}
        {tamura && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-slate-700 mb-3">Tamura Features</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">Coarseness</span>
                <div className="flex items-center gap-2">
                  <Badge variant="default">{tamura.coarseness.toFixed(2)}</Badge>
                  <div className="w-32 bg-slate-100 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${tamura.coarseness * 100}%` }}
                    />
                  </div>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">Contrast</span>
                <div className="flex items-center gap-2">
                  <Badge variant="default">{tamura.contrast.toFixed(2)}</Badge>
                  <div className="w-32 bg-slate-100 rounded-full h-2">
                    <div
                      className="bg-purple-500 h-2 rounded-full"
                      style={{ width: `${tamura.contrast * 100}%` }}
                    />
                  </div>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">Directionality</span>
                <div className="flex items-center gap-2">
                  <Badge variant="default">{tamura.directionality.toFixed(2)}</Badge>
                  <div className="w-32 bg-slate-100 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${tamura.directionality * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Gabor Features */}
        {gabor && (
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-3">Gabor Filter Responses</h4>
            <div className="grid grid-cols-4 gap-2">
              {gabor.map((value, index) => (
                <div key={index} className="text-center">
                  <div className="bg-slate-50 rounded-lg p-2 mb-1">
                    <div className="text-lg font-semibold text-slate-900">
                      {value.toFixed(2)}
                    </div>
                  </div>
                  <span className="text-xs text-slate-500">Filter {index + 1}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default TextureDescriptors;
