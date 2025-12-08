import React, { useState } from 'react';
import { Input, Dropdown, Button } from '../ui';

const FilterBar = ({ onFilterChange, categories = [] }) => {
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [showDetectionsOnly, setShowDetectionsOnly] = useState(false);
  
  const handleSearchChange = (e) => {
    const value = e.target.value;
    setSearch(value);
    onFilterChange({ search: value, category: selectedCategory, hasDetections: showDetectionsOnly });
  };
  
  const handleCategoryChange = (value) => {
    setSelectedCategory(value);
    onFilterChange({ search, category: value, hasDetections: showDetectionsOnly });
  };
  
  const handleToggleDetections = () => {
    const newValue = !showDetectionsOnly;
    setShowDetectionsOnly(newValue);
    onFilterChange({ search, category: selectedCategory, hasDetections: newValue });
  };
  
  const handleClearFilters = () => {
    setSearch('');
    setSelectedCategory('');
    setShowDetectionsOnly(false);
    onFilterChange({ search: '', category: '', hasDetections: false });
  };
  
  const categoryOptions = [
    { value: '', label: 'All Categories' },
    ...categories.map(cat => ({ value: cat, label: cat })),
  ];
  
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 mb-6">
      <div className="flex flex-wrap items-center gap-4">
        {/* Search */}
        <div className="flex-1 min-w-[200px]">
          <Input
            type="text"
            placeholder="Search by filename..."
            value={search}
            onChange={handleSearchChange}
            className="w-full"
          />
        </div>
        
        {/* Category Filter */}
        <div className="w-48">
          <Dropdown
            options={categoryOptions}
            value={selectedCategory}
            onChange={handleCategoryChange}
            placeholder="Filter by category"
          />
        </div>
        
        {/* Toggle Detections */}
        <Button
          variant={showDetectionsOnly ? 'primary' : 'secondary'}
          size="md"
          onClick={handleToggleDetections}
        >
          <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {showDetectionsOnly ? 'Detected Only' : 'All Images'}
        </Button>
        
        {/* Clear Filters */}
        {(search || selectedCategory || showDetectionsOnly) && (
          <Button variant="ghost" size="md" onClick={handleClearFilters}>
            Clear Filters
          </Button>
        )}
      </div>
    </div>
  );
};

export default FilterBar;
