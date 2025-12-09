import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button, Card, Spinner } from '../ui';

const UploadPanel = ({ onUpload }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState([]);
  const [warning, setWarning] = useState('');
  
  const onDrop = useCallback(async (acceptedFiles) => {
    // Only allow single image upload
    if (acceptedFiles.length > 1) {
      setWarning('⚠️ Please upload only ONE image at a time');
      setTimeout(() => setWarning(''), 3000);
      return;
    }

    if (acceptedFiles.length === 0) {
      return;
    }

    setUploading(true);
    setWarning('');
    
    // Initialize progress tracking for single file
    const progressList = [{
      name: acceptedFiles[0].name,
      progress: 0,
      status: 'uploading',
    }];
    setUploadProgress(progressList);
    
    try {
      // Simulate upload progress
      for (let p = 0; p <= 100; p += 20) {
        await new Promise(resolve => setTimeout(resolve, 100));
        setUploadProgress([{ 
          name: acceptedFiles[0].name, 
          progress: p, 
          status: 'uploading' 
        }]);
      }
      
      setUploadProgress([{ 
        name: acceptedFiles[0].name, 
        progress: 100, 
        status: 'complete' 
      }]);
      
      await onUpload(acceptedFiles[0]);
      
      // Clear progress after success
      setTimeout(() => {
        setUploadProgress([]);
        setUploading(false);
      }, 1000);
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadProgress([{ 
        name: acceptedFiles[0].name, 
        progress: 0, 
        status: 'error' 
      }]);
      setUploading(false);
    }
  }, [onUpload]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    },
    multiple: false,
    maxFiles: 1,
  });
  
  return (
    <Card className="mb-6">
      <Card.Body>
        {/* Warning Message */}
        {warning && (
          <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-800 text-sm font-medium">
            {warning}
          </div>
        )}
        
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition ${
            isDragActive
              ? 'border-indigo-500 bg-indigo-50'
              : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'
          }`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center space-y-4">
            <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center">
              <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div>
              <p className="text-lg font-medium text-slate-700">
                {isDragActive ? 'Drop file here' : 'Upload ONE image'}
              </p>
              <p className="text-sm text-slate-500 mt-1">Drag & drop or click to browse</p>
              <p className="text-xs text-red-500 mt-2">⚠️ Only 1 image allowed per upload</p>
            </div>
            <Button variant="secondary" size="sm">
              Browse Files
            </Button>
          </div>
        </div>
        
        {/* Upload Progress */}
        {uploadProgress.length > 0 && (
          <div className="mt-4 space-y-2">
            {uploadProgress.map((file, index) => (
              <div key={index} className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                <div className="flex-shrink-0">
                  {file.status === 'uploading' && <Spinner size="sm" />}
                  {file.status === 'complete' && (
                    <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  )}
                  {file.status === 'error' && (
                    <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-700 truncate">{file.name}</p>
                  {file.status === 'uploading' && (
                    <div className="mt-1 w-full bg-slate-200 rounded-full h-1.5">
                      <div
                        className="bg-indigo-600 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${file.progress}%` }}
                      />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default UploadPanel;
