import { useState } from 'react';
import { C1Component } from '@thesysai/genui-sdk';
import { BsExclamationTriangle, BsFileText, BsLink45Deg, BsCheck2Square, BsImages, BsStars } from 'react-icons/bs';
import { motion, AnimatePresence } from 'framer-motion';

import SourceCard from './SourceCard';
import StepsTimeline from './StepsTimeline';
import ImageGrid from './ImageGrid';
import ProcessingTimeline from './ProcessingTimeline';
import StreamingMarkdown from './StreamingMarkdown';

const MotionDiv = motion.div;

const ResponseContainer = ({ response }) => {
  const [activeTab, setActiveTab] = useState('Answer');

  const hasContent = response.streamingMarkdown || response.auiSpec;
  const warnings = response.providerWarnings || [];
  const isStreaming = response.streamingMarkdown
    && response.progress?.currentStage === 'synthesizing'
    && !response.isLoadedFromHistory;

  return (
    <div>
      <div className="user-prompt">
        {response.prompt}
      </div>

      <AnimatePresence mode="wait">
        {!hasContent && !response.error && (
          <MotionDiv
            key="timeline-view"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <ProcessingTimeline progress={response.progress} />
          </MotionDiv>
        )}

        {hasContent && (
          <MotionDiv
            key="content-view"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="ai-response-container">
              {(warnings.length > 0 || response.error) && (
                <div className="response-notice-stack">
                  {warnings.map((warning) => (
                    <div key={warning} className="response-notice warning">
                      <BsExclamationTriangle />
                      <span>{warning}</span>
                    </div>
                  ))}
                  {response.error && (
                    <div className="response-notice error">
                      <BsExclamationTriangle />
                      <span>{response.error}</span>
                    </div>
                  )}
                </div>
              )}

              <div className="tabs">
                <button
                  className={`tab ${activeTab === 'Answer' ? 'active' : ''}`}
                  onClick={() => setActiveTab('Answer')}
                >
                  <BsFileText /> Answer
                </button>

                {response.auiSpec && response.auiSpec.trim() && (
                  <button
                    className={`tab ${activeTab === 'Interactive' ? 'active' : ''}`}
                    onClick={() => setActiveTab('Interactive')}
                  >
                    <BsStars /> Interactive
                  </button>
                )}

                {response.sources?.length > 0 && (
                  <button
                    className={`tab ${activeTab === 'Sources' ? 'active' : ''}`}
                    onClick={() => setActiveTab('Sources')}
                  >
                    <BsLink45Deg /> Sources - {response.sources.length}
                  </button>
                )}

                {response.images?.length > 0 && (
                  <button
                    className={`tab ${activeTab === 'Images' ? 'active' : ''}`}
                    onClick={() => setActiveTab('Images')}
                  >
                    <BsImages /> Images - {response.images.length}
                  </button>
                )}

                {response.steps?.length > 0 && (
                  <button
                    className={`tab ${activeTab === 'Steps' ? 'active' : ''}`}
                    onClick={() => setActiveTab('Steps')}
                  >
                    <BsCheck2Square /> Steps
                  </button>
                )}
              </div>

              <div className="tab-content">
                {activeTab === 'Answer' && (
                  <StreamingMarkdown
                    content={response.streamingMarkdown}
                    isStreaming={isStreaming}
                    sources={response.sources || []}
                  />
                )}

                {activeTab === 'Interactive' && response.auiSpec && (
                  <C1Component c1Response={response.auiSpec} />
                )}

                {activeTab === 'Sources' && (
                  <div className="sources-grid">
                    {response.sources.map((src) => (
                      <SourceCard key={src.url || src.title} source={src} />
                    ))}
                  </div>
                )}

                {activeTab === 'Images' && (
                  <ImageGrid images={response.images} />
                )}

                {activeTab === 'Steps' && (
                  <StepsTimeline steps={response.steps} />
                )}
              </div>
            </div>
          </MotionDiv>
        )}

        {response.error && !hasContent && (
          <MotionDiv
            key="error-view"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            <div className="error-message">
              <BsExclamationTriangle />
              <span>{response.error}</span>
            </div>
          </MotionDiv>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ResponseContainer;
