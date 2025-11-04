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

  const { path, currentStage, queriesGenerated, sourcesFound, sourcesBeingScraped, totalScraped } = progress;


  if (currentStage === 'error') {
    return (
      <div className="processing-timeline error-state">
        <TimelineStep status="complete" title="Analysis complete" />
        <TimelineStep status="error" title="An error occurred during processing" />
      </div>
    );
  }

  // Determine the status of each major stage
  const analysisStatus = 'complete';
  const searchStatus = currentStage === 'searching' ? 'active' : (currentStage !== 'analyzing' ? 'complete' : 'pending');
  const readingStatus = currentStage === 'reading' ? 'active' : (currentStage === 'synthesizing' || currentStage === 'complete' ? 'complete' : 'pending');
  const synthesisStatus = currentStage === 'synthesizing' ? 'active' : 'pending';

  const isSearchPath = path === 'search_required';

  return (
    <div className="processing-timeline">
      {/* --- Analysis Step --- */}
      <TimelineStep status={analysisStatus} title={isSearchPath ? "Analyzing question" : "Understanding question"} />

      {/* --- Search/Direct Path Steps --- */}
      {isSearchPath ? (
        <>
          <TimelineStep status={searchStatus} title="Searching the web">
            {queriesGenerated.length > 0 && (
              <div className="sub-item-box"> {/* <-- The new box container */}
                <ul className="sub-item-list">
                  {queriesGenerated.map((query, i) => <li key={i}>"{query}"</li>)}
                </ul>
              </div>
            )}
          </TimelineStep>

          <TimelineStep status={readingStatus} title={readingStatus === 'complete' ? `Reviewed ${totalScraped} sources` : 'Reading sources'}>
            {sourcesFound > 0 && <p className="sub-item-counter">Found {sourcesFound} potential sources...</p>}
            {sourcesBeingScraped.length > 0 && (
              <div className="sub-item-box"> {/* <-- The new box container */}
                <ul className="sub-item-list reading-list">
                  {/* Conditionally render all items or just the first 3 */}
                  {(isReadingExpanded ? sourcesBeingScraped : sourcesBeingScraped.slice(0, 3)).map((domain, i) => (
                    <li key={i}>Reading {domain}</li>
                  ))}
                </ul>
                {/* The "Show More" button, which only appears if needed */}
                {!isReadingExpanded && sourcesBeingScraped.length > 3 && (
                  <button className="expand-button" onClick={() => setIsReadingExpanded(true)}>
                    and {sourcesBeingScraped.length - 3} more...
                  </button>
                )}
              </div>
            )}
          </TimelineStep>
        </>
      ) : (
        <TimelineStep status={synthesisStatus} title="Generating detailed response" />
      )}

      {/* --- Synthesis Step --- */}
      <TimelineStep status={synthesisStatus} title={isSearchPath ? "Generating answer" : "Formatting your answer"} />
    </div>
  );
};

export default ProcessingTimeline;