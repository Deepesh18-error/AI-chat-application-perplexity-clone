import React from 'react';

function SourceCard({ source }) {
  const hostname = new URL(source.url).hostname.replace('www.', '');
  return (
    <a href={source.url} target="_blank" rel="noopener noreferrer" className="source-card">
      <img
        src={`https://www.google.com/s2/favicons?domain=${hostname}&sz=32`}
        alt={`${hostname} favicon`}
        className="favicon"
      />
      <div className="source-info">
        <span className="source-title">{source.title || hostname}</span>
        <span className="source-hostname">{hostname}</span>
      </div>
    </a>
  );
}

export default SourceCard;