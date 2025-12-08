import React from 'react';

const Spinner = ({ size = 'md', className = '' }) => {
  const sizes = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-2',
    lg: 'w-12 h-12 border-3',
  };
  
  return (
    <div
      className={`inline-block border-indigo-600 border-t-transparent rounded-full animate-spin ${sizes[size]} ${className}`}
    />
  );
};

export default Spinner;
