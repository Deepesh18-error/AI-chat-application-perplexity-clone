import React, { useState } from 'react';
import HexagonalGrid from './landing/HexagonalGrid';
import ArgonCore from './landing/ArgonCore';
import './WelcomeScreen.css';

// The component now accepts all the props from App.jsx
function WelcomeScreen({ onExampleClick, prompt, setPrompt, handleSubmit, isLoading, forceWebSearch, 
  setForceWebSearch, isSpeechRecognitionSupported, isListening, handleMicClick }) {
  // This state now lives in the parent and controls the entire screen's effect
  const [isEnergized, setIsEnergized] = useState(false);

  return (
    // We apply the dynamic class here to affect the entire screen
    <div className={`welcome-container ${isEnergized ? 'is-energized' : ''}`}>
      <HexagonalGrid />
      <ArgonCore 
        // Pass down all the original props
        onExampleClick={onExampleClick}
        prompt={prompt}
        setPrompt={setPrompt}
        handleSubmit={handleSubmit}
        isLoading={isLoading}
        // Pass the state setter function down so the child can control the parent's state
        onFocusChange={setIsEnergized} 

        forceWebSearch={forceWebSearch}
        setForceWebSearch={setForceWebSearch}
        isSpeechRecognitionSupported={isSpeechRecognitionSupported}
        isListening={isListening}
        handleMicClick={handleMicClick}
      />
    </div>
  );
}

export default WelcomeScreen;