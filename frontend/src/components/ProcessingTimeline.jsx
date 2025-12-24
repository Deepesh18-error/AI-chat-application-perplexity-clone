import React, { useState } from 'react';
import { BsCheckCircleFill, BsDot } from 'react-icons/bs';
import { AiOutlineLoading3Quarters } from 'react-icons/ai'; 

// A small, reusable helper component for each step in the timeline
const TimelineStep = ({ status, title, children }) => {
  // REMOVED: useState and the faulty "if (!progress)" check.

  const getIcon = () => {
    switch (status) {
      case 'complete':
        return <BsCheckCircleFill className="icon-complete" />;
      case 'active':
        return <AiOutlineLoading3Quarters className="icon-active spinner" />;
      default: // 'pending'
        return <BsDot className="icon-pending" />;
    }
  };

  return (
    <div className={`timeline-step ${status}`}>
      <div className="timeline-icon">{getIcon()}</div>
      <div className="timeline-content">
        <p className="timeline-title">{title}</p>
        {children && <div className="timeline-children">{children}</div>}
      </div>
    </div>
  );
};

// The main timeline component
const ProcessingTimeline = ({ progress }) => {
  const [isReadingExpanded, setIsReadingExpanded] = useState(false);
  if (!progress) return null;

  const { path, currentStage, queriesGenerated, sourcesRetrieved } = progress;

  // 1. UPDATED STAGE LOGIC
  const analysisStatus = 'complete';
  // Retrieval is active if we are 'searching' or 'retrieving'
  const retrievalStatus = (currentStage === 'searching' || currentStage === 'retrieving') 
    ? 'active' 
    : (currentStage === 'synthesizing' || currentStage === 'complete' ? 'complete' : 'pending');
  
  const synthesisStatus = currentStage === 'synthesizing' ? 'active' : (currentStage === 'complete' ? 'complete' : 'pending');

  const isSearchPath = path === 'search_required';

  return (
    <div className="processing-timeline">
      <TimelineStep status={analysisStatus} title={isSearchPath ? "Analyzing your question" : "Understanding question"} />

      {isSearchPath ? (
        <TimelineStep 
          status={retrievalStatus} 
          title={retrievalStatus === 'complete' ? `Read ${sourcesRetrieved} sources` : "Retrieving information"}
        >
          {/* Show the search queries being used */}
          {queriesGenerated.length > 0 && (
            <div className="sub-item-box">
              <ul className="sub-item-list">
                {queriesGenerated.map((query, i) => <li key={i}>Searching for "{query}"...</li>)}
              </ul>
              {/* Show the live counter of sources retrieved in real-time */}
              {sourcesRetrieved > 0 && (
                <p className="sub-item-counter" style={{marginTop: '10px', color: '#3b82f6'}}>
                  ✓ Found and processed {sourcesRetrieved} sources
                </p>
              )}
            </div>
          )}
        </TimelineStep>
      ) : (
        <TimelineStep status={synthesisStatus} title="Processing direct answer" />
      )}

      <TimelineStep status={synthesisStatus} title="Generating final answer" />
    </div>
  );
};

export default ProcessingTimeline;