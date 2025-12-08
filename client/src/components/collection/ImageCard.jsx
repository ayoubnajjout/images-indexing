import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Badge, Button } from '../ui';

const ImageCard = ({ image, onDelete, onDetect }) => {
  const navigate = useNavigate();
  
  const handleViewDetails = () => {
    navigate(`/image/${image.id}`);
  };
  
  const handleDownload = () => {
    // Create a temporary anchor element to trigger download
    const link = document.createElement('a');
    link.href = image.url;
    link.download = image.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  return (
    <Card className="overflow-hidden group">
      {/* Image */}
      <div className="relative aspect-video overflow-hidden bg-slate-100">
        <img
          src={image.url}
          alt={image.filename}
          className="w-full h-full object-cover transition duration-300 group-hover:scale-105"
        />
        
        {/* Overlay with actions (visible on hover) */}
        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition duration-300 flex items-center justify-center opacity-0 group-hover:opacity-100">
          <div className="flex gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleViewDetails}
              className="bg-white"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleDownload}
              className="bg-white"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </Button>
            <Button
              variant="danger"
              size="sm"
              onClick={() => onDelete(image.id)}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </Button>
          </div>
        </div>
        
        {/* Object count badge */}
        {image.objectCount > 0 && (
          <div className="absolute top-2 right-2">
            <Badge variant="primary">
              {image.objectCount} {image.objectCount === 1 ? 'object' : 'objects'}
            </Badge>
          </div>
        )}
      </div>
      
      {/* Info */}
      <Card.Body className="py-3">
        <h3 className="font-medium text-slate-900 truncate mb-1">{image.filename}</h3>
        <p className="text-sm text-slate-500">{image.uploadDate}</p>
        
        {/* Actions */}
        <div className="mt-3 flex gap-2">
          {image.objectCount === 0 ? (
            <Button
              variant="primary"
              size="sm"
              onClick={() => onDetect(image.id)}
              className="w-full"
            >
              Detect Objects
            </Button>
          ) : (
            <Button
              variant="outline"
              size="sm"
              onClick={handleViewDetails}
              className="w-full"
            >
              View Details
            </Button>
          )}
        </div>
      </Card.Body>
    </Card>
  );
};

export default ImageCard;
