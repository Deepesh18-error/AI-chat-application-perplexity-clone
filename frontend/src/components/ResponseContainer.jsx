import React, { useState } from 'react';
import { C1Component } from '@thesysai/genui-sdk';
import SourceCard from './SourceCard';
import StepsTimeline from './StepsTimeline';
import { BsFileText, BsLink45Deg, BsCheck2Square, BsImages, BsStars } from 'react-icons/bs';
import ImageGrid from './ImageGrid';
import ProcessingTimeline from './ProcessingTimeline';
import StreamingMarkdown from './StreamingMarkdown';
import { motion, AnimatePresence } from 'framer-motion';

const ResponseContainer = ({ response, isLastTurn }) => {
  const [activeTab, setActiveTab] = useState('Answer');

  const hasContent = response.streamingMarkdown || response.auiSpec;
  const isStreaming = response.streamingMarkdown && 
  response.progress?.currentStage === 'synthesizing' && 
  !response.isLoadedFromHistory;

  return (
    <div>
      {/* User's Prompt Bubble */}
      <div className="user-prompt">
        {response.prompt}
      </div>

      <AnimatePresence mode="wait">

        {/* STATE 1: No content yet — show ProcessingTimeline */}
        {!hasContent && !response.error && (
          <motion.div
            key="timeline-view"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <ProcessingTimeline progress={response.progress} />
          </motion.div>
        )}

        {/* STATE 2 & 3: Has content — show tabbed UI (markdown + optional Thesys tab) */}
        {hasContent && !response.error && (
          <motion.div
            key="content-view"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="ai-response-container">

              {/* Tabs */}
              <div className="tabs">
                <button
                  className={`tab ${activeTab === 'Answer' ? 'active' : ''}`}
                  onClick={() => setActiveTab('Answer')}
                >
                  <BsFileText /> Answer
                </button>

                {/* Interactive tab — only shown once Thesys finishes */}
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
                    <BsLink45Deg /> Sources · {response.sources.length}
                  </button>
                )}

                {response.images?.length > 0 && (
                  <button
                    className={`tab ${activeTab === 'Images' ? 'active' : ''}`}
                    onClick={() => setActiveTab('Images')}
                  >
                    <BsImages /> Images · {response.images.length}
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

              {/* Tab Content */}
              <div className="tab-content">

                {/* Answer tab — always the live markdown */}
                {activeTab === 'Answer' && (
                  <StreamingMarkdown
                    content={response.streamingMarkdown}
                    isStreaming={isStreaming}
                    sources={response.sources || []}
                  />
                )}

                {/* Interactive tab — Thesys C1 rendered UI */}
                {activeTab === 'Interactive' && response.auiSpec && (
                  <C1Component c1Response={response.auiSpec} />
                )}

                {activeTab === 'Sources' && (
                  <div className="sources-grid">
                    {response.sources.map((src, i) => (
                      <SourceCard key={i} source={src} />
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
          </motion.div>
        )}

        {/* Error state */}
        {response.error && (
          <motion.div
            key="error-view"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            <div className="error-message">{response.error}</div>
          </motion.div>
        )}

      </AnimatePresence>
    </div>
  );
};

export default ResponseContainer;