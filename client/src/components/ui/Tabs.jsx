import React, { useState } from 'react';

const Tabs = ({ tabs, defaultTab = 0, className = '' }) => {
  const [activeTab, setActiveTab] = useState(defaultTab);
  
  return (
    <div className={className}>
      <div className="border-b border-slate-200">
        <nav className="flex space-x-8">
          {tabs.map((tab, index) => (
            <button
              key={index}
              onClick={() => setActiveTab(index)}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition ${
                activeTab === index
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
      <div className="py-6">
        {tabs[activeTab].content}
      </div>
    </div>
  );
};

export default Tabs;
