import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from './components/layout';
import { 
  CollectionPage, 
  ImageDetailPage, 
  DescriptorsPage, 
  SearchPage, 
  CreatePage,
  Search3DPage
} from './pages';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<CollectionPage />} />
          <Route path="/image/:id" element={<ImageDetailPage />} />
          <Route path="/descriptors" element={<DescriptorsPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/search-3d" element={<Search3DPage />} />
          <Route path="/create" element={<CreatePage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
