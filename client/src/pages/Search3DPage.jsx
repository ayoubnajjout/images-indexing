/**
 * 3D Model Search Page
 * Content-based retrieval for 3D models using shape descriptors
 */

import { useState, useEffect } from 'react';
import { model3DService } from '../services';
import Model3DCard from '../components/search/Model3DCard';
import Model3DViewer from '../components/search/Model3DViewer';
import DescriptorVisualization from '../components/search/DescriptorVisualization';

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Progress Bar Component for indexing operations with real-time updates
 */
const IndexingProgressBar = ({ progress }) => {
  if (!progress) return null;
  
  const { current = 0, total = 0, status, currentFile, category, newlyIndexed = 0, failed = 0, toIndex = 0 } = progress;
  const percentage = total > 0 ? Math.round((current / total) * 100) : 0;
  
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-2xl p-6 w-[420px] max-w-[90vw]">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Indexing 3D Models</h3>
          <div className="animate-pulse w-3 h-3 bg-green-500 rounded-full"></div>
        </div>
        
        {/* Progress bar */}
        <div className="h-5 bg-gray-200 rounded-full overflow-hidden mb-3 relative">
          <div 
            className="h-full bg-gradient-to-r from-blue-500 via-blue-400 to-green-500 transition-all duration-300 ease-out"
            style={{ width: `${percentage}%` }}
          />
          <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white drop-shadow">
            {percentage}%
          </span>
        </div>
        
        {/* Stats grid */}
        <div className="grid grid-cols-3 gap-3 mb-3 text-center">
          <div className="bg-gray-50 rounded p-2">
            <div className="text-lg font-bold text-gray-800">{current}</div>
            <div className="text-xs text-gray-500">Processed</div>
          </div>
          <div className="bg-green-50 rounded p-2">
            <div className="text-lg font-bold text-green-600">{newlyIndexed}</div>
            <div className="text-xs text-gray-500">New</div>
          </div>
          <div className="bg-red-50 rounded p-2">
            <div className="text-lg font-bold text-red-600">{failed}</div>
            <div className="text-xs text-gray-500">Failed</div>
          </div>
        </div>
        
        {/* Status text */}
        <div className="bg-gray-50 rounded p-2">
          <div className="text-sm text-gray-700 truncate font-medium">
            {status || 'Initializing...'}
          </div>
          {category && currentFile && (
            <div className="text-xs text-gray-500 mt-1 truncate">
              {category} / {currentFile}
            </div>
          )}
        </div>
        
        {/* Total info */}
        <div className="mt-3 text-xs text-gray-400 text-center">
          Total: {total} models {toIndex > 0 && `(${toIndex} to index)`}
        </div>
      </div>
    </div>
  );
};

/**
 * Search Result Item with 2D thumbnail display
 */
const SearchResultItem = ({ result, rank, onSelect, onCompare }) => {
  const [imageError, setImageError] = useState(false);
  const thumbnailUrl = result.thumbnailUrl ? model3DService.getThumbnailUrl(result.thumbnailUrl) : null;
  
  const displayName = result.name || result.filename?.replace('.obj', '');
  const similarity = result.similarity || 0;
  
  return (
    <div
      className="bg-gray-50 rounded-lg p-3 hover:bg-gray-100 cursor-pointer transition-colors"
      onClick={() => onSelect(result)}
    >
      <div className="flex gap-3">
        {/* Thumbnail */}
        <div className="w-20 h-20 bg-gray-200 rounded overflow-hidden flex-shrink-0">
          {thumbnailUrl && !imageError ? (
            <img
              src={thumbnailUrl}
              alt={displayName}
              className="w-full h-full object-contain"
              onError={() => setImageError(true)}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-gray-400">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} 
                  d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
            </div>
          )}
        </div>
        
        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div className="min-w-0">
              <p className="font-medium text-sm text-gray-800 truncate">{displayName}</p>
              <p className="text-xs text-gray-600">{result.category}</p>
            </div>
            
            {/* Similarity badge */}
            <div className={`flex-shrink-0 ml-2 px-2 py-0.5 rounded-full text-xs font-medium
              ${similarity >= 0.8 ? 'bg-green-100 text-green-700' :
                similarity >= 0.6 ? 'bg-yellow-100 text-yellow-700' :
                'bg-gray-100 text-gray-700'}`}
            >
              {(similarity * 100).toFixed(1)}%
            </div>
          </div>
          
          {/* Rank and compare */}
          <div className="flex items-center justify-between mt-1">
            <p className="text-xs text-gray-500">
              Rank #{rank} {result.numVertices ? `• ${result.numVertices.toLocaleString()} verts` : ''}
            </p>
            {onCompare && result._id && (
              <button
                onClick={(e) => { e.stopPropagation(); onCompare(result); }}
                className="text-xs text-blue-600 hover:text-blue-800"
              >
                Compare
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Loading Spinner
 */
const Spinner = ({ size = 'md', className = '' }) => {
  const sizeClasses = { sm: 'h-4 w-4', md: 'h-8 w-8', lg: 'h-12 w-12' };
  return (
    <div className={`animate-spin rounded-full border-b-2 border-blue-600 ${sizeClasses[size]} ${className}`} />
  );
};

// ============================================================================
// Main Component
// ============================================================================

const Search3DPage = () => {
  // Core state
  const [categories, setCategories] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [selectedModel, setSelectedModel] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState(null);
  
  // Data state
  const [stats, setStats] = useState(null);
  const [descriptorInfo, setDescriptorInfo] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  
  // Pagination & search options
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [searchLimit, setSearchLimit] = useState(20);
  const [searchScope, setSearchScope] = useState('all');
  
  // Descriptor visualization
  const [selectedModelDescriptors, setSelectedModelDescriptors] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [showDescriptorPanel, setShowDescriptorPanel] = useState(false);
  
  // Progress tracking
  const [indexingProgress, setIndexingProgress] = useState(null);
  const [showDescriptorInfo, setShowDescriptorInfo] = useState(false);

  // ============================================================================
  // Data Loading
  // ============================================================================

  useEffect(() => {
    loadCategories();
    loadStats();
    loadDescriptorInfo();
  }, []);

  useEffect(() => {
    loadModels();
  }, [selectedCategory, page]);

  const loadCategories = async () => {
    try {
      const response = await model3DService.getCategories();
      if (response.success) setCategories(response.categories);
    } catch (err) {
      console.error('Failed to load categories:', err);
    }
  };

  const loadModels = async () => {
    setLoading(true);
    try {
      const response = await model3DService.getModels({
        category: selectedCategory || undefined,
        page,
        limit: 20
      });
      if (response.success) {
        setModels(response.models);
        setTotalPages(response.totalPages || 1);
      }
    } catch (err) {
      setError('Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await model3DService.getStats();
      if (response.success) setStats(response.stats);
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  };

  const loadDescriptorInfo = async () => {
    try {
      const response = await model3DService.getDescriptorInfo();
      if (response.success) setDescriptorInfo(response);
    } catch (err) {
      console.error('Failed to load descriptor info:', err);
    }
  };

  const loadModelDescriptors = async (modelId) => {
    try {
      const response = await model3DService.getModelDescriptors(modelId);
      if (response.success) {
        setSelectedModelDescriptors({
          model: response.model,
          descriptors: response.descriptors
        });
      }
    } catch (err) {
      console.error('Failed to load descriptors:', err);
    }
  };

  // ============================================================================
  // Event Handlers
  // ============================================================================

  const handleModelSelect = (model) => {
    setSelectedModel(model);
    setSearchResults([]);
    setUploadedFile(null);
    setSelectedModelDescriptors(null);
    setComparisonResult(null);
    
    if (model._id && model.isIndexed) {
      loadModelDescriptors(model._id);
    }
  };

  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
    setPage(1);
  };

  const handleCompareDescriptors = async (resultModel) => {
    if (!selectedModel?._id || !resultModel?._id) return;
    
    try {
      const response = await model3DService.compareDescriptors(selectedModel._id, resultModel._id);
      if (response.success) setComparisonResult(response);
    } catch (err) {
      console.error('Failed to compare:', err);
    }
  };

  const handleSearch = async (model) => {
    if (!model) return;
    
    setSearching(true);
    setSelectedModel(model);
    setError(null);
    
    try {
      const categoryFilter = (searchScope === 'category' && selectedCategory) ? selectedCategory : undefined;
      
      let response;
      if (model._id) {
        response = await model3DService.searchByModelId(model._id, { limit: searchLimit, category: categoryFilter });
      } else {
        const modelPath = model.filepath || 
          `${import.meta.env.VITE_MODELS_3D_PATH || '/app/3d-data/3D Models'}/${model.category}/${model.filename}`;
        response = await model3DService.searchByPath(modelPath, { limit: searchLimit, category: categoryFilter });
      }
      
      if (response.success) {
        setSearchResults(response.results || []);
      } else {
        setError('Search failed');
      }
    } catch (err) {
      setError('Search failed: ' + err.message);
    } finally {
      setSearching(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file || !file.name.endsWith('.obj')) {
      setError('Please select a .obj file');
      return;
    }
    
    setUploadedFile(file);
    setSelectedModel({ name: file.name, isUploaded: true });
    setSearching(true);
    setError(null);
    
    try {
      const categoryFilter = (searchScope === 'category' && selectedCategory) ? selectedCategory : undefined;
      const response = await model3DService.searchByUpload(file, { limit: searchLimit, category: categoryFilter });
      
      if (response.success) {
        setSearchResults(response.results || []);
      } else {
        setError('Search failed');
      }
    } catch (err) {
      setError('Search failed: ' + err.message);
    } finally {
      setSearching(false);
    }
  };

  // ============================================================================
  // Indexing Operations
  // ============================================================================

  const handleIndexModel = async (model) => {
    setIndexing(true);
    setError(null);
    
    try {
      const response = await model3DService.indexModel({
        category: model.category,
        filename: model.filename
      });
      
      if (response.success) {
        loadModels();
        loadStats();
      }
    } catch (err) {
      setError('Indexing failed: ' + err.message);
    } finally {
      setIndexing(false);
    }
  };

  const handleIndexCategory = async (category) => {
    if (!category) return;
    
    setIndexing(true);
    setError(null);
    
    try {
      const response = await model3DService.indexCategory(category);
      
      if (response.success) {
        alert(`Indexed ${response.indexed} models in ${category}`);
        loadStats();
        loadModels();
      } else {
        setError('Indexing failed');
      }
    } catch (err) {
      setError('Indexing failed: ' + err.message);
    } finally {
      setIndexing(false);
    }
  };

  const handleIndexAll = async () => {
    if (!confirm('Index all 3D models? Already indexed models will be skipped.')) return;
    
    setIndexing(true);
    setError(null);
    setIndexingProgress({ current: 0, total: 0, status: 'Connecting...' });
    
    try {
      // Use SSE streaming for real-time progress
      const response = await model3DService.indexAllCategoriesWithProgress(5, (progress) => {
        setIndexingProgress(progress);
      });
      
      if (response.success) {
        setIndexingProgress({
          current: response.totalFiles,
          total: response.totalFiles,
          status: 'Complete!'
        });
        
        await new Promise(r => setTimeout(r, 800));
        
        const msg = `Indexing Complete!\n\n` +
          `Total: ${response.totalFiles}\n` +
          `New: ${response.newlyIndexed || 0}\n` +
          `Skipped: ${response.skipped || 0}\n` +
          `Failed: ${response.failed || 0}`;
        alert(msg);
        
        loadStats();
        loadModels();
      } else {
        setError('Indexing failed');
      }
    } catch (err) {
      setError('Indexing failed: ' + err.message);
    } finally {
      setIndexing(false);
      setIndexingProgress(null);
    }
  };

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Progress Modal */}
      <IndexingProgressBar progress={indexingProgress} />
      
      {/* Header */}
      <header className="bg-white border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">3D Model Search</h1>
            <p className="text-gray-600 text-sm mt-1">
              Content-based retrieval using shape descriptors (D1, D2, D3, D4, A3)
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            {stats && (
              <div className="text-sm text-gray-600">
                <span className="font-medium">{stats.indexedModels}</span> / {stats.totalModels} indexed
              </div>
            )}
            
            <button
              onClick={handleIndexAll}
              disabled={indexing}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 font-medium text-sm flex items-center gap-2"
            >
              {indexing ? (
                <>
                  <Spinner size="sm" className="border-white" />
                  Indexing...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                  Index All
                </>
              )}
            </button>
            
            <button
              onClick={() => setShowDescriptorInfo(!showDescriptorInfo)}
              className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
            >
              {showDescriptorInfo ? 'Hide' : 'Show'} Info
            </button>
          </div>
        </div>
        
        {/* Descriptor Info Panel */}
        {showDescriptorInfo && descriptorInfo && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <h3 className="font-medium text-blue-800 mb-2">Shape Descriptors</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
              {descriptorInfo.descriptors?.slice(0, 5).map((desc, i) => (
                <div key={i} className="bg-white p-2 rounded shadow-sm">
                  <div className="font-medium text-sm text-gray-800">{desc.name}</div>
                  <div className="text-xs text-gray-600 mt-1">{desc.description}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </header>

      {/* Main Layout */}
      <div className="flex h-[calc(100vh-140px)]">
        {/* Left Sidebar */}
        <aside className="w-64 bg-white border-r p-4 overflow-y-auto">
          {/* Upload */}
          <section className="mb-6">
            <h3 className="font-medium text-gray-800 mb-2">Upload Query Model</h3>
            <label className="block cursor-pointer">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-blue-500 transition-colors">
                <svg className="w-8 h-8 mx-auto text-gray-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <span className="text-sm text-gray-600">
                  {uploadedFile ? uploadedFile.name : 'Upload .obj file'}
                </span>
              </div>
              <input type="file" accept=".obj" onChange={handleFileUpload} className="hidden" />
            </label>
          </section>

          {/* Categories */}
          <section className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium text-gray-800">Categories</h3>
              {selectedCategory && (
                <button
                  onClick={() => handleIndexCategory(selectedCategory)}
                  disabled={indexing}
                  className="text-xs text-blue-600 hover:text-blue-800 disabled:opacity-50"
                >
                  Index
                </button>
              )}
            </div>
            
            <nav className="space-y-1">
              <button
                onClick={() => handleCategoryChange('')}
                className={`w-full text-left px-3 py-2 rounded text-sm transition-colors
                  ${!selectedCategory ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-100 text-gray-700'}`}
              >
                All Categories
              </button>
              
              {categories.map(cat => (
                <button
                  key={cat.name}
                  onClick={() => handleCategoryChange(cat.name)}
                  className={`w-full text-left px-3 py-2 rounded text-sm transition-colors flex justify-between
                    ${selectedCategory === cat.name ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-100 text-gray-700'}`}
                >
                  <span className="truncate">{cat.name}</span>
                  <span className="text-xs text-gray-400">{cat.count}</span>
                </button>
              ))}
            </nav>
          </section>

          {/* Search Options */}
          <section>
            <h3 className="font-medium text-gray-800 mb-2">Search Options</h3>
            
            <label className="block text-sm text-gray-600 mb-1">Scope</label>
            <select
              value={searchScope}
              onChange={(e) => setSearchScope(e.target.value)}
              className="w-full px-3 py-2 border rounded text-sm mb-3"
            >
              <option value="all">All Categories</option>
              <option value="category">Current Category</option>
            </select>
            
            <label className="block text-sm text-gray-600 mb-1">Limit</label>
            <select
              value={searchLimit}
              onChange={(e) => setSearchLimit(parseInt(e.target.value))}
              className="w-full px-3 py-2 border rounded text-sm"
            >
              {[10, 20, 50, 100].map(n => (
                <option key={n} value={n}>{n} results</option>
              ))}
            </select>
            
            {searchScope === 'all' && (
              <p className="mt-2 text-xs text-green-600">✓ Searching all categories</p>
            )}
          </section>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-4 overflow-y-auto">
          {error && (
            <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg flex justify-between items-center">
              {error}
              <button onClick={() => setError(null)} className="text-red-500 hover:text-red-700">✕</button>
            </div>
          )}

          {loading ? (
            <div className="flex items-center justify-center h-64">
              <Spinner size="lg" />
            </div>
          ) : (
            <>
              <div className="mb-4 flex items-center justify-between">
                <h2 className="font-medium text-gray-800">
                  {selectedCategory || 'All Models'} ({models.length})
                </h2>
                
                {totalPages > 1 && (
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setPage(p => Math.max(1, p - 1))}
                      disabled={page === 1}
                      className="px-3 py-1 text-sm border rounded disabled:opacity-50"
                    >
                      Prev
                    </button>
                    <span className="text-sm text-gray-600">
                      {page} / {totalPages}
                    </span>
                    <button
                      onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                      disabled={page === totalPages}
                      className="px-3 py-1 text-sm border rounded disabled:opacity-50"
                    >
                      Next
                    </button>
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
                {models.map((model, idx) => (
                  <Model3DCard
                    key={model._id || `${model.category}-${model.filename}-${idx}`}
                    model={model}
                    selected={selectedModel?.filename === model.filename}
                    onSelect={handleModelSelect}
                    onSearch={handleSearch}
                  />
                ))}
              </div>

              {models.length === 0 && (
                <div className="text-center text-gray-500 py-12">
                  No models found
                </div>
              )}
            </>
          )}
        </main>

        {/* Right Sidebar */}
        <aside className="w-80 bg-white border-l p-4 overflow-y-auto">
          {selectedModel ? (
            <>
              {/* Query Model */}
              <section className="mb-6">
                <h3 className="font-medium text-gray-800 mb-3">Query Model</h3>
                
                <div className="bg-gray-50 rounded-lg p-3">
                  {selectedModel.modelUrl && (
                    <Model3DViewer
                      modelUrl={model3DService.getModelFileUrl(selectedModel.modelUrl)}
                      width={260}
                      height={200}
                      className="mb-3"
                    />
                  )}
                  
                  <div className="text-sm">
                    <p className="font-medium text-gray-800 truncate">
                      {selectedModel.name || selectedModel.filename}
                    </p>
                    <p className="text-gray-600">{selectedModel.category}</p>
                    {selectedModel.numVertices && (
                      <p className="text-gray-500 text-xs mt-1">
                        {selectedModel.numVertices.toLocaleString()} vertices
                      </p>
                    )}
                  </div>
                  
                  <div className="mt-3 flex gap-2">
                    <button
                      onClick={() => handleSearch(selectedModel)}
                      disabled={searching}
                      className="flex-1 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
                    >
                      {searching ? 'Searching...' : 'Find Similar'}
                    </button>
                    
                    {!selectedModel.isIndexed && !selectedModel.isUploaded && (
                      <button
                        onClick={() => handleIndexModel(selectedModel)}
                        disabled={indexing}
                        className="px-3 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-700 disabled:opacity-50"
                      >
                        Index
                      </button>
                    )}
                  </div>
                  
                  {selectedModel.isIndexed && selectedModel._id && (
                    <button
                      onClick={() => setShowDescriptorPanel(!showDescriptorPanel)}
                      className="mt-2 w-full py-1.5 text-xs bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                    >
                      {showDescriptorPanel ? 'Hide' : 'View'} Descriptors
                    </button>
                  )}
                </div>
              </section>
              
              {/* Descriptors Panel */}
              {showDescriptorPanel && selectedModelDescriptors && (
                <section className="mb-6">
                  <DescriptorVisualization
                    descriptors={selectedModelDescriptors.descriptors}
                    modelInfo={selectedModelDescriptors.model}
                    comparison={comparisonResult?.comparison}
                    compact={true}
                  />
                </section>
              )}
              
              {/* Comparison Result */}
              {comparisonResult && (
                <section className="mb-6 p-3 bg-blue-50 rounded-lg">
                  <h4 className="text-sm font-medium text-blue-800 mb-2">
                    Comparison Result
                  </h4>
                  <div className="text-center">
                    <div className={`text-2xl font-bold ${
                      comparisonResult.overallSimilarity >= 0.8 ? 'text-green-600' :
                      comparisonResult.overallSimilarity >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {(comparisonResult.overallSimilarity * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-blue-600">Overall Similarity</div>
                  </div>
                  <button
                    onClick={() => setComparisonResult(null)}
                    className="mt-2 w-full text-xs text-blue-600 hover:text-blue-800"
                  >
                    Clear
                  </button>
                </section>
              )}

              {/* Search Results */}
              {searching ? (
                <div className="flex justify-center py-8">
                  <Spinner />
                </div>
              ) : searchResults.length > 0 ? (
                <section>
                  <h3 className="font-medium text-gray-800 mb-3">
                    Similar Models ({searchResults.length})
                  </h3>
                  <div className="space-y-3">
                    {searchResults.map((result, idx) => (
                      <SearchResultItem 
                        key={result._id || `${result.category}-${result.filename}-${idx}`}
                        result={result}
                        rank={idx + 1}
                        onSelect={handleModelSelect}
                        onCompare={handleCompareDescriptors}
                      />
                    ))}
                  </div>
                </section>
              ) : (
                <div className="text-center text-gray-500 py-8">
                  <p>Select a model and click "Find Similar"</p>
                  <p className="text-sm mt-1">or upload a .obj file</p>
                </div>
              )}
            </>
          ) : (
            <div className="text-center text-gray-500 py-8">
              <svg className="w-16 h-16 mx-auto text-gray-300 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} 
                  d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
              <p>Select a 3D model to view details</p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
};

export default Search3DPage;
