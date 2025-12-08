import React from 'react';
import { Card, Button, Badge } from '../ui';

const TransformControls = ({ transformations, onTransformationChange, onReset }) => {
  const handleScaleChange = (e) => {
    onTransformationChange({ ...transformations, scale: parseInt(e.target.value) });
  };
  
  const handleRotate = (degrees) => {
    const currentRotation = transformations.rotate || 0;
    onTransformationChange({ 
      ...transformations, 
      rotate: (currentRotation + degrees) % 360 
    });
  };
  
  const handleFlip = (axis) => {
    if (axis === 'horizontal') {
      onTransformationChange({ 
        ...transformations, 
        flipH: !transformations.flipH 
      });
    } else {
      onTransformationChange({ 
        ...transformations, 
        flipV: !transformations.flipV 
      });
    }
  };
  
  return (
    <Card>
      <Card.Header>
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-slate-900">Transform Controls</h3>
          <Button variant="ghost" size="sm" onClick={onReset}>
            Reset All
          </Button>
        </div>
      </Card.Header>
      <Card.Body className="space-y-6">
        {/* Scale */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-slate-700">Scale</label>
            <Badge variant="default">{transformations.scale || 100}%</Badge>
          </div>
          <input
            type="range"
            min="25"
            max="200"
            step="5"
            value={transformations.scale || 100}
            onChange={handleScaleChange}
            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
          />
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>25%</span>
            <span>100%</span>
            <span>200%</span>
          </div>
        </div>
        
        {/* Rotate */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <label className="text-sm font-medium text-slate-700">Rotate</label>
            <Badge variant="default">{transformations.rotate || 0}°</Badge>
          </div>
          <div className="grid grid-cols-4 gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => handleRotate(-90)}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2M3 12l6.414 6.414a2 2 0 001.414.586H19a2 2 0 002-2V7a2 2 0 00-2-2h-8.172a2 2 0 00-1.414.586L3 12z" />
              </svg>
              -90°
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => handleRotate(-45)}
            >
              -45°
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => handleRotate(45)}
            >
              +45°
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => handleRotate(90)}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12l-6.414-6.414a2 2 0 00-1.414-.586H5a2 2 0 00-2 2v10a2 2 0 002 2h8.172a2 2 0 001.414-.586L21 12z" />
              </svg>
              +90°
            </Button>
          </div>
        </div>
        
        {/* Flip */}
        <div>
          <label className="text-sm font-medium text-slate-700 mb-3 block">Flip</label>
          <div className="grid grid-cols-2 gap-2">
            <Button
              variant={transformations.flipH ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => handleFlip('horizontal')}
            >
              <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
              </svg>
              Horizontal
            </Button>
            <Button
              variant={transformations.flipV ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => handleFlip('vertical')}
            >
              <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
              </svg>
              Vertical
            </Button>
          </div>
        </div>
        
        {/* Crop (simplified) */}
        <div>
          <label className="text-sm font-medium text-slate-700 mb-3 block">Aspect Ratio</label>
          <div className="grid grid-cols-3 gap-2">
            <Button variant="secondary" size="sm">Original</Button>
            <Button variant="secondary" size="sm">1:1</Button>
            <Button variant="secondary" size="sm">16:9</Button>
            <Button variant="secondary" size="sm">4:3</Button>
            <Button variant="secondary" size="sm">3:2</Button>
            <Button variant="secondary" size="sm">Free</Button>
          </div>
        </div>
      </Card.Body>
    </Card>
  );
};

export default TransformControls;
