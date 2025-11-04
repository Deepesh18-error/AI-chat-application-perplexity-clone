import React, { useState } from 'react';

import { C1Component } from '@thesysai/genui-sdk'; 
import SourceCard from './SourceCard';
import StepsTimeline from './StepsTimeline';
import { BsFileText, BsLink45Deg, BsCheck2Square, BsImages } from 'react-icons/bs';
import ImageGrid from './ImageGrid';
import ProcessingTimeline from './ProcessingTimeline';
import { motion, AnimatePresence } from 'framer-motion';


const ResponseContainer = ({ response , isLastTurn  }) => {
  const [activeTab, setActiveTab] = useState('Answer');
  const isFocusedView = activeTab !== 'Answer' && !isLastTurn;

    const animationVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
  };

  return (
    <div>
      {/*  1. The User's Prompt Bubble  */}
      <div className="user-prompt">
        {response.prompt}
      </div>

      {/*  2. The AI's Response Area  */}
      <AnimatePresence mode="wait">
 {response.auiSpec && response.auiSpec.trim() && !response.error ? (
    // IF we have the final answer, render the answer view
    <motion.div
      key="answer-view"
      variants={animationVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={{ duration: 0.4 }}
    >
          <div className={`ai-response-container ${isFocusedView ? 'focused-view' : ''}`}>
            {/*  Tabs Section  */}
            <div className="tabs">
              <button
                className={`tab ${activeTab === 'Answer' ? 'active' : ''}`}
                onClick={() => setActiveTab('Answer')}
              >
                <BsFileText /> Answer
              </button>

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

              {response.steps && (
                <button
                  className={`tab ${activeTab === 'Steps' ? 'active' : ''}`}
                  onClick={() => setActiveTab('Steps')}
                >
                  <BsCheck2Square /> Steps
                </button>
              )}
            </div>

            {/*  Tab Content  */}
            <div className="tab-content">
              {activeTab === 'Answer' && <C1Component c1Response={response.auiSpec} />}

              {activeTab === 'Sources' && (
                <div className="sources-grid">
                  {response.sources.map((src, i) => (
                    <SourceCard key={i} source={src} />
                  ))}
                </div>
              )}

              {activeTab === 'Images' && <ImageGrid images={response.images} />}

              {activeTab === 'Steps' && <StepsTimeline steps={response.steps} />}
            </div>
          </div>
        </motion.div>
      ) : (
        // ELSE, if the answer is not ready, render the timeline
        <motion.div
          key="timeline-view"
          variants={animationVariants}
          initial="initial"
          animate="animate"
          exit="exit"
          transition={{ duration: 0.4 }}
        >
          <ProcessingTimeline progress={response.progress} />
        </motion.div>
      )}
    </AnimatePresence>

    </div>
  );
};

export default ResponseContainer;