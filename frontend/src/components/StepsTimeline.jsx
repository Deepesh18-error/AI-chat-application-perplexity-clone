function StepsTimeline({ steps, isLoading }) {
  if (steps.length === 0 && !isLoading) return <p>No steps to show.</p>;
  if (steps.length === 0 && isLoading) return <p className="step-item current">Initializing...</p>;

  return (
    <div className="steps-timeline">
      {steps.map((step, index) => {
        const isSearchQuery = step.startsWith('Searching for:');
        const message = isSearchQuery ? step.substring(15) : step;
        const isCurrent = isLoading && index === steps.length - 1;

        if (isSearchQuery) {
          return (
            <div key={`${message}-${index}`} className={`step-item search-query ${isCurrent ? 'current' : 'done'}`}>
              <span className="query-icon">Search</span>
              <code className="query-text">{message}</code>
            </div>
          );
        }

        return (
          <p key={`${message}-${index}`} className={`step-item ${isCurrent ? 'current' : 'done'}`}>
            {message}
          </p>
        );
      })}
    </div>
  );
}

export default StepsTimeline;
