import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from './components/layout';
import { 
  CollectionPage, 
  ImageDetailPage, 
  DescriptorsPage, 
  SearchPage, 
  CreatePage 
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
          <Route path="/create" element={<CreatePage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
