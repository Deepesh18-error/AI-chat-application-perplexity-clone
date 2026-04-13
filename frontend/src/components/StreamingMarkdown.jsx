import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Citation badge with tooltip showing source title + URL
const CitationBadge = ({ num, sources }) => {
  const [isHovered, setIsHovered] = useState(false);
  
  // Citations are 1-indexed, sources array is 0-indexed
  const source = sources[num - 1];

  return (
    <span 
      className="citation-wrapper"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <sup className="citation-badge">{num}</sup>
      
      {isHovered && source && (
        <span className="citation-tooltip">
          <span className="citation-tooltip-num">Source {num}</span>
          <span className="citation-tooltip-title">{source.title}</span>
          <a 
            href={source.url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="citation-tooltip-url"
            onClick={(e) => e.stopPropagation()}
          >
            {new URL(source.url).hostname.replace('www.', '')}
          </a>
        </span>
      )}
    </span>
  );
};

// Processes text and replaces [1], [2][3] patterns with CitationBadge components
const processCitations = (text, sources) => {
  if (typeof text !== 'string') return text;
  const parts = text.split(/(\[\d+\](?:\[\d+\])*)/g);
  
  return parts.map((part, index) => {
    if (/^\[\d+\]/.test(part)) {
      const numbers = [...part.matchAll(/\[(\d+)\]/g)].map(m => parseInt(m[1]));
      return (
        <span key={index} className="citation-group">
          {numbers.map((num) => (
            <CitationBadge key={num} num={num} sources={sources} />
          ))}
        </span>
      );
    }
    return part;
  });
};

// Wraps any child text nodes and processes citations inside them
const CitationText = ({ children, sources }) => {
  if (typeof children === 'string') {
    return <>{processCitations(children, sources)}</>;
  }
  if (Array.isArray(children)) {
    return (
      <>
        {children.map((child, i) =>
          typeof child === 'string'
            ? <React.Fragment key={i}>{processCitations(child, sources)}</React.Fragment>
            : <React.Fragment key={i}>{child}</React.Fragment>
        )}
      </>
    );
  }
  return <>{children}</>;
};

// Build markdown component map — all receive sources via closure
const buildComponents = (sources) => ({
  table: ({ children }) => (
    <div className="md-table-wrapper">
      <table className="md-table">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="md-thead">{children}</thead>,
  tbody: ({ children }) => <tbody>{children}</tbody>,

  tr: ({ children }) => <tr className="md-tr">{children}</tr>,

  th: ({ children }) => <th className="md-th"><CitationText sources={sources}>{children}</CitationText></th>,

  td: ({ children }) => <td className="md-td"><CitationText sources={sources}>{children}</CitationText></td>,

  p:  ({ children }) => <p className="md-p"><CitationText sources={sources}>{children}</CitationText></p>,

  li: ({ children }) => <li className="md-li"><CitationText sources={sources}>{children}</CitationText></li>,

  strong: ({ children }) => <strong className="md-strong"><CitationText sources={sources}>{children}</CitationText></strong>,

  h1: ({ children }) => <h1 className="md-h1"><CitationText sources={sources}>{children}</CitationText></h1>,

  h2: ({ children }) => <h2 className="md-h2"><CitationText sources={sources}>{children}</CitationText></h2>,

  
  h3: ({ children }) => <h3 className="md-h3"><CitationText sources={sources}>{children}</CitationText></h3>,


  blockquote: ({ children }) => <blockquote className="md-blockquote">{children}</blockquote>,
  code: ({ inline, children }) =>
    inline
      ? <code className="md-code-inline">{children}</code>
      : <pre className="md-code-block"><code>{children}</code></pre>,
});

function StreamingMarkdown({ content, isStreaming, sources = [] }) {
  const components = buildComponents(sources);

  return (
    <div className="streaming-markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
      {isStreaming && <span className="streaming-cursor" />}
    </div>
  );
}

export default StreamingMarkdown;