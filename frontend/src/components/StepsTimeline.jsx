import React from 'react';

function StepsTimeline({ steps, isLoading }) {
  if (steps.length === 0 && !isLoading) return <p>No steps to show.</p>;
  if (steps.length === 0 && isLoading) return <p className="step-item current">Initializing...</p>;
  
  return (
    <div className="steps-timeline">
      {steps.map((step, index) => (
        <p key={index} className={`step-item ${isLoading && index === steps.length - 1 ? 'current' : 'done'}`}>
          {step}
        </p>
      ))}
    </div>
  );
}

export default StepsTimeline;