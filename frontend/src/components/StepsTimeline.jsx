// src/components/StepsTimeline.jsx - ENHANCED VERSION

import React from 'react';

function StepsTimeline({ steps, isLoading }) {
  if (steps.length === 0 && !isLoading) return <p>No steps to show.</p>;
  if (steps.length === 0 && isLoading) return <p className="step-item current">Initializing...</p>;
  
  return (
    <div className="steps-timeline">
      {steps.map((step, index) => {
        // Check if the current step is a search query
        const isSearchQuery = step.startsWith("Searching for:");

        // Get the main message, removing the prefix for cleaner display
        const message = isSearchQuery ? step.substring(15) : step;

        const isCurrent = isLoading && index === steps.length - 1;

        if (isSearchQuery) {
          return (
            <div key={index} className={`step-item search-query ${isCurrent ? 'current' : 'done'}`}>
              <span className="query-icon">🔍</span>
              <code className="query-text">{message}</code>
            </div>
          );
        } else {
          return (
            <p key={index} className={`step-item ${isCurrent ? 'current' : 'done'}`}>
              {message}
            </p>
          );
        }
      })}
    </div>
  );
}

export default StepsTimeline;