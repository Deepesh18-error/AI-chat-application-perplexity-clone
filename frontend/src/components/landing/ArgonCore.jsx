import React from 'react';
import { FiLoader, FiGlobe } from 'react-icons/fi';
import { BsMicFill } from 'react-icons/bs';
import TextareaAutosize from 'react-textarea-autosize';


function ArgonCore({ onExampleClick, prompt, setPrompt, handleSubmit, isLoading, onFocusChange, forceWebSearch, setForceWebSearch, 
  isSpeechRecognitionSupported, isListening, handleMicClick }) {

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      // The handleSubmit prop is already wrapped in handleFormSubmit
      handleFormSubmit(e); 
    }
  };
  const handleFormSubmit = (e) => {
    e.preventDefault();
    handleSubmit(e);
  };

  return (
    <div className="argon-core">
      {/* --- ALL ELECTRONS NOW LIVE HERE AT THE TOP LEVEL --- */}
      {/* Existing Electrons */}
      <div className="electron electron-1"></div>
      <div className="electron electron-2"></div>
      <div className="electron electron-3"></div>
      <div className="electron electron-4"></div>
      {/* New, Smaller Electrons */}
      <div className="electron electron-5 electron-small"></div>
      <div className="electron electron-6 electron-small"></div>
      <div className="electron electron-7 electron-small"></div>
      <div className="electron electron-8 electron-tiny"></div>
      <div className="electron electron-9 electron-tiny"></div>
      <div className="electron electron-10 electron-tiny"></div>

      {/* --- THIS NEW CONTAINER WRAPS ALL YOUR UI CONTENT --- */}
      <div className="argon-content">
  {/* SECTION 1: The Header */}
        <div className="landing-header">
          <h1>MEV</h1>
          <div className="keywords">
            <span className="keyword-item">Interactive UI</span>
            <span className="keyword-item">LLM Response</span>
            <span className="keyword-item">Web Search</span>
            <span className="keyword-item">Context Memory</span>
          </div>
        </div>

        {/* NEW MESSAGE AREA */}
        <p className="tagline">What can I help with?</p>
 

        {/* SECTION 2: The Search Form */}
        <form onSubmit={handleFormSubmit} className="landing-prompt-form">
          <TextareaAutosize
                  className="landing-input"
                  placeholder="Ask me anything..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onFocus={() => onFocusChange(true)}
                  onBlur={() => onFocusChange(false)}
                  disabled={isLoading}
                  onKeyDown={handleKeyDown} // <-- Add the handler
                  rows={1}
                  maxRows={8}
                />
          
          {/* NEW: A container for all the action buttons at the bottom */}
          <div className="landing-actions-bar">
            <div className="landing-actions-left">
              <button
                type="button"
                onClick={() => setForceWebSearch(prev => !prev)}
                className={`action-btn web-search-toggle ${forceWebSearch ? 'active' : ''}`}
                title="Force Web Search"
              >
                <FiGlobe /> {/* <-- This is the new SVG icon */}
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

        {/* SECTION 3: Example Prompts */}
        <div className="example-prompts-landing">
          <button onClick={() => onExampleClick("What is Bernoulli's principle?")} disabled={isLoading}>
            What is Bernoulli's principle?
          </button>
          <button onClick={() => onExampleClick("Create a short story...")} disabled={isLoading}>
            Create a short story...
          </button>
          <button onClick={() => onExampleClick("Who is the current CEO of OpenAI?")} disabled={isLoading}>
            Who is the current CEO of OpenAI?
          </button>
        </div>
      </div>
    </div>
  );
}

export default ArgonCore;