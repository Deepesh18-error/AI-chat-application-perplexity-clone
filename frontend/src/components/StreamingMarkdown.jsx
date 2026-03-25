import React from 'react';
import ReactMarkdown from 'react-markdown';

function StreamingMarkdown({ content, isStreaming }) {
  return (
    <div className="streaming-markdown">
      <ReactMarkdown>{content}</ReactMarkdown>
      {isStreaming && <span className="streaming-cursor" />}
    </div>
  );
}

export default StreamingMarkdown;