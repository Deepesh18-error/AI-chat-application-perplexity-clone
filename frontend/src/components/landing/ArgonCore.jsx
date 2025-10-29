import React from 'react';
import { FiLoader } from 'react-icons/fi';
import { BsMicFill } from 'react-icons/bs';

function ArgonCore({ onExampleClick, prompt, setPrompt, handleSubmit, isLoading, onFocusChange, forceWebSearch, setForceWebSearch, 
  isSpeechRecognitionSupported, isListening, handleMicClick }) {
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
          <h1>MEV</h1> {/* I've updated the name to match your latest image */}
          <p className="tagline">Catalyze a query. Emit an answer.</p>
          <div className="keywords">
            <span className="keyword-item">Interactive UI</span>
            <span className="keyword-item">LLM Response</span>
            <span className="keyword-item">Web Search</span>
            <span className="keyword-item">Context Memory</span>
          </div>
        </div>

        {/* SECTION 2: The Search Form */}
        <form onSubmit={handleFormSubmit} className="landing-prompt-form">

          <button
            type="button"
            onClick={() => setForceWebSearch(prev => !prev)}
            className={`web-search-toggle ${forceWebSearch ? 'active' : ''}`}
            title="Force Web Search"
          >
            🌐
          </button>

          {isSpeechRecognitionSupported && (
            <button
              type="button"
              // --- START OF MODIFICATION ---
              className={`mic-button ${isListening ? 'active' : ''}`}
              onClick={handleMicClick}
              // --- END OF MODIFICATION ---
              title="Use Microphone"
            >
              <BsMicFill />
            </button>
          )}


          <input
            type="text"
            className="landing-input"
            placeholder="Ask me anything..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onFocus={() => onFocusChange(true)}
            onBlur={() => onFocusChange(false)}
            disabled={isLoading}
          />
          <button type="submit" className="landing-submit-btn" disabled={isLoading || !prompt.trim()}>
            {isLoading ? <FiLoader className="spinner" /> : 'Ask'}
          </button>
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