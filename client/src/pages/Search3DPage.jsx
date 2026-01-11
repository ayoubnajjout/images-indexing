/**
 * 3D Model Search Page
 * Content-based retrieval for 3D models using shape descriptors
 */

import { useState, useEffect, useCallback } from 'react';
import { model3DService } from '../services';
import Model3DCard from '../components/search/Model3DCard';
import Model3DViewer from '../components/search/Model3DViewer';

const Search3DPage = () => {
  // State
  const [categories, setCategories] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [selectedModel, setSelectedModel] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [descriptorInfo, setDescriptorInfo] = useState(null);
  const [indexing, setIndexing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [searchLimit, setSearchLimit] = useState(10);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [showDescriptorInfo, setShowDescriptorInfo] = useState(false);

  // Load initial data
  useEffect(() => {
    loadCategories();
    loadStats();
    loadDescriptorInfo();
  }, []);

  // Load models when category changes
  useEffect(() => {
    loadModels();
  }, [selectedCategory, page]);

  const loadCategories = async () => {
    try {
      const response = await model3DService.getCategories();
      if (response.success) {
        setCategories(response.categories);
      }
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
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await model3DService.getStats();
      if (response.success) {
        setStats(response.stats);
      }
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  };

  const loadDescriptorInfo = async () => {
    try {
      const response = await model3DService.getDescriptorInfo();
      if (response.success) {
        setDescriptorInfo(response);
      }
    } catch (err) {
      console.error('Failed to load descriptor info:', err);
    }
  };

  const handleModelSelect = (model) => {
    setSelectedModel(model);
    setSearchResults([]);
    setUploadedFile(null);
  };

  const handleSearch = async (model) => {
    if (!model) return;
    
    setSearching(true);
    setSelectedModel(model);
    setError(null);
    
    try {
      let response;
      
      if (model._id) {
        // Search using indexed model
        response = await model3DService.searchByModelId(model._id, {
          limit: searchLimit,
          category: selectedCategory || undefined
        });
      } else {
        // Search using file path
        const modelPath = model.filepath || 
          `${import.meta.env.VITE_MODELS_3D_PATH || '/app/3d-data/3D Models'}/${model.category}/${model.filename}`;
        
        response = await model3DService.searchByPath(modelPath, {
          limit: searchLimit,
          category: selectedCategory || undefined
        });
      }
      
      if (response.success) {
        setSearchResults(response.results || []);
      } else {
        setError('Search failed');
      }
    } catch (err) {
      setError('Search failed: ' + err.message);
      console.error(err);
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
      const response = await model3DService.searchByUpload(file, {
        limit: searchLimit,
        category: selectedCategory || undefined
      });
      
      if (response.success) {
        setSearchResults(response.results || []);
      } else {
        setError('Search failed');
      }
    } catch (err) {
      setError('Search failed: ' + err.message);
      console.error(err);
    } finally {
      setSearching(false);
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
      console.error(err);
    } finally {
      setIndexing(false);
    }
  };

  const handleIndexAll = async () => {
    if (!confirm('This will index all 3D models from all categories. This may take a long time. Continue?')) {
      return;
    }
    
    setIndexing(true);
    setError(null);
    
    try {
      const response = await model3DService.indexAllCategories();
      
      if (response.success) {
        const message = `Successfully indexed ${response.indexed} models from ${response.totalCategories} categories!\n` +
                       `Total: ${response.totalFiles} files\n` +
                       `Success: ${response.indexed}\n` +
                       `Failed: ${response.failed}`;
        alert(message);
        loadStats();
        loadModels();
      } else {
        setError('Indexing failed');
      }
    } catch (err) {
      setError('Indexing failed: ' + err.message);
      console.error(err);
    } finally {
      setIndexing(false);
    }
  };

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

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">3D Model Search</h1>
            <p className="text-gray-600 text-sm mt-1">
              Content-based retrieval using shape descriptors (D1, D2, D3, D4, A3)
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Stats */}
            {stats && (
              <div className="text-sm text-gray-600">
                <span className="font-medium">{stats.indexedModels}</span> / {stats.totalModels} indexed
              </div>
            )}
            
            {/* Index All Button */}
            <button
              onClick={handleIndexAll}
              disabled={indexing}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium text-sm flex items-center gap-2"
            >
              {indexing ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
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
            
            {/* Descriptor Info Button */}
            <button
              onClick={() => setShowDescriptorInfo(!showDescriptorInfo)}
              className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
            >
              {showDescriptorInfo ? 'Hide' : 'Show'} Descriptor Info
            </button>
          </div>
        </div>
        
        {/* Descriptor Info Panel */}
        {showDescriptorInfo && descriptorInfo && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <h3 className="font-medium text-blue-800 mb-2">Shape Descriptors Used</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
              {descriptorInfo.descriptors?.slice(0, 5).map((desc, i) => (
                <div key={i} className="bg-white p-2 rounded shadow-sm">
                  <div className="font-medium text-sm text-gray-800">{desc.name}</div>
                  <div className="text-xs text-gray-600 mt-1">{desc.description}</div>
                </div>
              ))}
            </div>
            <p className="mt-2 text-xs text-blue-700">
              OpenGL rendering: {descriptorInfo.opengl_enabled ? 'Enabled' : 'Disabled'}
            </p>
          </div>
        )}
      </div>

      <div className="flex h-[calc(100vh-140px)]">
        {/* Left Sidebar - Categories & Upload */}
        <div className="w-64 bg-white border-r p-4 overflow-y-auto">
          {/* Upload Section */}
          <div className="mb-6">
            <h3 className="font-medium text-gray-800 mb-2">Upload Query Model</h3>
            <label className="block">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-blue-500 transition-colors">
                <svg className="w-8 h-8 mx-auto text-gray-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <span className="text-sm text-gray-600">
                  {uploadedFile ? uploadedFile.name : 'Upload .obj file'}
                </span>
              </div>
              <input
                type="file"
                accept=".obj"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
          </div>

          {/* Categories */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium text-gray-800">Categories</h3>
              {selectedCategory && (
                <button
                  onClick={() => handleIndexCategory(selectedCategory)}
                  disabled={indexing}
                  className="text-xs text-blue-600 hover:text-blue-800 disabled:opacity-50"
                >
                  {indexing ? 'Indexing...' : 'Index All'}
                </button>
              )}
            </div>
            
            <div className="space-y-1">
              <button
                onClick={() => { setSelectedCategory(''); setPage(1); }}
                className={`w-full text-left px-3 py-2 rounded text-sm transition-colors
                  ${!selectedCategory ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-100 text-gray-700'}`}
              >
                All Categories
              </button>
              
              {categories.map(cat => (
                <button
                  key={cat.name}
                  onClick={() => { setSelectedCategory(cat.name); setPage(1); }}
                  className={`w-full text-left px-3 py-2 rounded text-sm transition-colors flex justify-between
                    ${selectedCategory === cat.name ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-100 text-gray-700'}`}
                >
                  <span className="truncate">{cat.name}</span>
                  <span className="text-xs text-gray-400">{cat.count}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Search Options */}
          <div className="mt-6">
            <h3 className="font-medium text-gray-800 mb-2">Search Options</h3>
            <label className="block text-sm text-gray-600 mb-1">
              Results Limit
            </label>
            <select
              value={searchLimit}
              onChange={(e) => setSearchLimit(parseInt(e.target.value))}
              className="w-full px-3 py-2 border rounded text-sm"
            >
              <option value={5}>5 results</option>
              <option value={10}>10 results</option>
              <option value={20}>20 results</option>
              <option value={50}>50 results</option>
            </select>
          </div>
        </div>

        {/* Main Content - Model Grid */}
        <div className="flex-1 p-4 overflow-y-auto">
          {error && (
            <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">
              {error}
            </div>
          )}

          {loading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
          ) : (
            <>
              <div className="mb-4 flex items-center justify-between">
                <h2 className="font-medium text-gray-800">
                  {selectedCategory || 'All Models'} ({models.length} shown)
                </h2>
                
                {/* Pagination */}
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
                      Page {page} of {totalPages}
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
                  No models found in this category
                </div>
              )}
            </>
          )}
        </div>

        {/* Right Sidebar - Selected Model & Results */}
        <div className="w-80 bg-white border-l p-4 overflow-y-auto">
          {selectedModel ? (
            <>
              {/* Selected Model Info */}
              <div className="mb-6">
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
                        {selectedModel.numVertices?.toLocaleString()} vertices
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
                </div>
              </div>

              {/* Search Results */}
              {searching ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                </div>
              ) : searchResults.length > 0 ? (
                <div>
                  <h3 className="font-medium text-gray-800 mb-3">
                    Similar Models ({searchResults.length})
                  </h3>
                  
                  <div className="space-y-3">
                    {searchResults.map((result, idx) => (
                      <div
                        key={result._id || `${result.category}-${result.filename}-${idx}`}
                        className="bg-gray-50 rounded-lg p-3 hover:bg-gray-100 cursor-pointer"
                        onClick={() => handleModelSelect(result)}
                      >
                        <div className="flex gap-3">
                          {/* Thumbnail */}
                          <div className="w-16 h-16 bg-gray-200 rounded overflow-hidden flex-shrink-0">
                            {result.thumbnailUrl ? (
                              <img
                                src={model3DService.getThumbnailUrl(result.thumbnailUrl)}
                                alt={result.name}
                                className="w-full h-full object-contain"
                              />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center text-gray-400">
                                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
                                <p className="font-medium text-sm text-gray-800 truncate">
                                  {result.name || result.filename?.replace('.obj', '')}
                                </p>
                                <p className="text-xs text-gray-600">{result.category}</p>
                              </div>
                              
                              {/* Similarity Score */}
                              <div className="flex-shrink-0 ml-2">
                                <div className={`px-2 py-0.5 rounded-full text-xs font-medium
                                  ${result.similarity >= 0.8 ? 'bg-green-100 text-green-700' :
                                    result.similarity >= 0.6 ? 'bg-yellow-100 text-yellow-700' :
                                    'bg-gray-100 text-gray-700'}`}
                                >
                                  {(result.similarity * 100).toFixed(1)}%
                                </div>
                              </div>
                            </div>
                            
                            {/* Rank */}
                            <p className="text-xs text-gray-500 mt-1">
                              Rank #{idx + 1}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-500 py-8">
                  <p>Select a model and click "Find Similar"</p>
                  <p className="text-sm mt-1">or upload a .obj file to search</p>
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
              <p className="text-sm mt-1">and search for similar models</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Search3DPage;
