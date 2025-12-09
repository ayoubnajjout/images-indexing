import React, { useState } from 'react';
import { Input, Dropdown, Button } from '../ui';

const FilterBar = ({ onFilterChange }) => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  
  const handleCategoryChange = (value) => {
    setSelectedCategory(value);
    onFilterChange({ category: value });
  };
  
  const categoryOptions = [
    { value: 'all', label: 'All Images' },
    { value: 'uploaded', label: 'Uploaded' },
    { value: 'edited', label: 'Edited' },
    { value: 'categories', label: 'Categories' },
  ];
  
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 mb-6">
      <div className="flex items-center gap-4">
        <span className="text-sm font-medium text-slate-700">Filter by:</span>
        
        {/* Category Filter */}
        <div className="flex gap-2">
          {categoryOptions.map(option => (
            <Button
              key={option.value}
              variant={selectedCategory === option.value ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => handleCategoryChange(option.value)}
            >
              {option.label}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FilterBar;
