import React, { useState } from 'react';

import { C1Component } from '@thesysai/genui-sdk'; 
import SourceCard from './SourceCard';
import StepsTimeline from './StepsTimeline';
import { BsFileText, BsLink45Deg, BsCheck2Square, BsImages } from 'react-icons/bs';
import ImageGrid from './ImageGrid';

function ResponseContainer({ response }) {
  const [activeTab, setActiveTab] = useState('answer');
  const isSearch = response.sources && response.sources.length > 0;
  const hasImages = response.images && response.images.length > 0;

  return (
    // --- REMOVE THE REDUNDANT ThemeProvider WRAPPER FROM THIS FILE ---
    <div className="response-container">
      <p className="response-prompt">
        {response.prompt}
      </p>

      <div className="tabs">
        <button onClick={() => setActiveTab('answer')} className={activeTab === 'answer' ? 'active' : ''}>
          <BsFileText /> Answer
        </button>
        {isSearch && (
          <button onClick={() => setActiveTab('sources')} className={activeTab === 'sources' ? 'active' : ''}>
            <BsLink45Deg /> Sources · {response.sources.length}
          </button>
        )}

        {hasImages && (
          <button onClick={() => setActiveTab('images')} className={activeTab === 'images' ? 'active' : ''}>
            <BsImages /> Images · {response.images.length}
          </button>
        )}

        
        <button onClick={() => setActiveTab('steps')} className={activeTab === 'steps' ? 'active' : ''}>
          <BsCheck2Square /> Steps
        </button>
      </div>

      <div className="tab-content">
        {activeTab === 'answer' && (
          <div className="response-completion">
            {/* --- THIS LOGIC IS NOW CORRECT AND FINAL --- */}
            {/* It checks if the auiSpec object exists and renders it with the C1Component */}
            {response.auiSpec ? (
              <C1Component c1Response={response.auiSpec} />
            ) : (
              // If the spec hasn't arrived yet but we are loading, show a cursor
              response.isLoading && <span className="blinking-cursor">|</span>
            )}
            {/* --- END OF THESYS IMPLEMENTATION --- */}
          </div>
        )}
        {activeTab === 'sources' && isSearch && (
          <div className="sources-grid">
            {response.sources.map((source, index) => (
              <SourceCard key={index} source={source} />
            ))}
          </div>
        )}

        {activeTab === 'images' && hasImages && (
          <ImageGrid images={response.images} />
        )}

        {activeTab === 'steps' && (
          <StepsTimeline steps={response.steps} isLoading={response.isLoading}/>
        )}
      </div>
       {response.error && <div className="error-message">Error: {response.error}</div>}
    </div>
  );
}

export default ResponseContainer;