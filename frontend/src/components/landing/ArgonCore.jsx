import React from 'react';import { FiLoader, FiGlobe } from 'react-icons/fi';
import { BsMicFill } from 'react-icons/bs';
import TextareaAutosize from 'react-textarea-autosize';

function ArgonCore({ 
  onExampleClick, 
  prompt, 
  setPrompt, 
  handleSubmit, 
  isLoading, 
  forceWebSearch, 
  setForceWebSearch, 
  isSpeechRecognitionSupported, 
  isListening, 
  handleMicClick
}) {

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleFormSubmit(e); 
    }
  };
  
  const handleFormSubmit = (e) => {
    e.preventDefault();
    handleSubmit(e);
  };

  return (
    <div className="argon-core">
      <div className="argon-content">
        <div className="landing-header">
          <h1>ARGON</h1>
          <div className="keywords">
            <span className="keyword-item">Interactive UI</span>
            <span className="keyword-item">LLM Response</span>
            <span className="keyword-item">Web Search</span>
            <span className="keyword-item">Context Memory</span>
          </div>
        </div>

        <div className="landing-interactive-area">
          <p className="tagline">What can I help with?</p>
  
          <form 
            onSubmit={handleFormSubmit} 
            className="landing-prompt-form"
          >
            <TextareaAutosize
              className="landing-input"
              placeholder="Ask me anything..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={isLoading}
              onKeyDown={handleKeyDown}
              rows={1}
              maxRows={8}
            />
            
            <div className="landing-actions-bar">
              <div className="landing-actions-left">
                <button
                  type="button"
                  onClick={() => setForceWebSearch(prev => !prev)}
                  className={`action-btn web-search-toggle ${forceWebSearch ? 'active' : ''}`}
                  title="Force Web Search"
                >
                  <FiGlobe />
                </button>

                {isSpeechRecognitionSupported && (
                  <button
                    type="button"
                    className={`action-btn mic-button ${isListening ? 'active' : ''}`}
                    onClick={handleMicClick}
                    title="Use Microphone"
                  >
                    <BsMicFill />
                  </button>
                )}
              </div>

              <div className="landing-actions-right">
                <button type="submit" className="submit-btn" disabled={isLoading || !prompt.trim()}>
                  {isLoading ? <FiLoader className="spinner" /> : 'Ask'}
                </button>
              </div>
            </div>
          </form>

          <div className="example-prompts-landing">
            <button onClick={() => onExampleClick("What is theory of relativity?")} disabled={isLoading}>
              What is theory of relativity?
            </button>
            <button onClick={() => onExampleClick("Give me code of...")} disabled={isLoading}>
              Give me code of...
            </button>
            <button onClick={() => onExampleClick("Who is the current CEO of OpenAI?")} disabled={isLoading}>
              Who is the current CEO of OpenAI?
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ArgonCore;