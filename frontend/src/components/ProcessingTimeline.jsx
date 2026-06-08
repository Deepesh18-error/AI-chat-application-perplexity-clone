import { BsCheckCircleFill, BsDot } from 'react-icons/bs';
import { AiOutlineLoading3Quarters } from 'react-icons/ai';

const TimelineStep = ({ status, title, children }) => {
  const getIcon = () => {
    switch (status) {
      case 'complete':
        return <BsCheckCircleFill className="icon-complete" />;
      case 'active':
        return <AiOutlineLoading3Quarters className="icon-active spinner" />;
      default:
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

const ProcessingTimeline = ({ progress }) => {
  if (!progress) return null;

  const {
    path,
    currentStage,
    queriesGenerated = [],
    sourcesRetrieved = 0,
  } = progress;

  const analysisStatus = 'complete';
  const retrievalStatus = (currentStage === 'searching' || currentStage === 'retrieving')
    ? 'active'
    : (currentStage === 'synthesizing' || currentStage === 'complete' ? 'complete' : 'pending');
  const synthesisStatus = currentStage === 'synthesizing'
    ? 'active'
    : (currentStage === 'complete' ? 'complete' : 'pending');

  const isSearchPath = path === 'search_required';

  return (
    <div className="processing-timeline">
      <TimelineStep status={analysisStatus} title={isSearchPath ? 'Analyzing your question' : 'Understanding question'} />

      {isSearchPath ? (
        <TimelineStep
          status={retrievalStatus}
          title={retrievalStatus === 'complete' ? `Read ${sourcesRetrieved} sources` : 'Retrieving information'}
        >
          {queriesGenerated.length > 0 && (
            <div className="sub-item-box">
              <ul className="sub-item-list">
                {queriesGenerated.map((query) => <li key={query}>Searching for "{query}"...</li>)}
              </ul>
            </div>
          )}
          {sourcesRetrieved > 0 && (
            <p className="sub-item-counter">
              Found and processed {sourcesRetrieved} sources
            </p>
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
