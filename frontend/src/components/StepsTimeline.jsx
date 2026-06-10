import {
  BsCheck2,
  BsCpu,
  BsGlobe2,
  BsLightningCharge,
  BsSearch,
  BsStars,
} from 'react-icons/bs';

const STEP_DETAILS = [
  {
    match: 'searching the web',
    title: 'Web Search',
    detail: 'Looking across live sources for fresh context.',
    icon: BsGlobe2,
  },
  {
    match: 'found quick results',
    title: 'Quick Results',
    detail: 'Pulled an instant summary before deeper synthesis.',
    icon: BsLightningCharge,
  },
  {
    match: 'synthesizing',
    title: 'Answer Synthesis',
    detail: 'Combining source snippets into a cited response.',
    icon: BsCpu,
  },
  {
    match: 'generating interactive ui',
    title: 'Interactive View',
    detail: 'Preparing the optional visual response layer.',
    icon: BsStars,
  },
  {
    match: 'generating answer',
    title: 'Direct Answer',
    detail: 'Using the model directly without live retrieval.',
    icon: BsCpu,
  },
];

const getStepMeta = (message) => {
  const normalized = message.toLowerCase();
  const matched = STEP_DETAILS.find((item) => normalized.includes(item.match));

  if (matched) return matched;

  if (normalized.startsWith('searching for:')) {
    return {
      title: 'Search Query',
      detail: 'A focused query generated for retrieval.',
      icon: BsSearch,
    };
  }

  return {
    title: 'Processing Step',
    detail: 'Completed during this response run.',
    icon: BsCheck2,
  };
};

function StepsTimeline({ steps = [], isLoading }) {
  if (steps.length === 0 && !isLoading) {
    return (
      <div className="steps-empty-state">
        <BsCheck2 />
        <span>No steps to show yet.</span>
      </div>
    );
  }

  if (steps.length === 0 && isLoading) {
    return (
      <div className="steps-timeline steps-polished">
        <div className="step-card current">
          <div className="step-node"><BsCpu /></div>
          <div className="step-content">
            <span className="step-eyebrow">Running</span>
            <h3>Initializing</h3>
            <p>Preparing the response pipeline.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="steps-timeline steps-polished">
      <div className="steps-summary-strip">
        <span>{steps.length} completed step{steps.length === 1 ? '' : 's'}</span>
        <strong>Execution Trace</strong>
      </div>

      {steps.map((step, index) => {
        const isSearchQuery = step.startsWith('Searching for:');
        const message = isSearchQuery ? step.substring(15) : step;
        const isCurrent = isLoading && index === steps.length - 1;
        const meta = getStepMeta(step);
        const Icon = meta.icon;

        return (
          <div key={`${message}-${index}`} className={`step-card ${isCurrent ? 'current' : 'done'}`}>
            <div className="step-node">
              <Icon />
            </div>
            <div className="step-content">
              <div className="step-header-line">
                <span className="step-eyebrow">Step {index + 1}</span>
                <span className="step-status">{isCurrent ? 'Running' : 'Done'}</span>
              </div>
              <h3>{meta.title}</h3>
              <p>{meta.detail}</p>
              {isSearchQuery ? (
                <code className="step-query-text">{message}</code>
              ) : (
                <span className="step-raw-message">{message}</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default StepsTimeline;
