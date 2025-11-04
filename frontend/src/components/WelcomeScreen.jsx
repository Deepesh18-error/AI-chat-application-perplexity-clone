import React from 'react';
import StarField from './landing/StarField';
import ArgonCore from './landing/ArgonCore';
import './WelcomeScreen.css';

function WelcomeScreen({ 
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
  return (
    <div className="welcome-container">
      <StarField />
      <ArgonCore 
        onExampleClick={onExampleClick}
        prompt={prompt}
        setPrompt={setPrompt}
        handleSubmit={handleSubmit}
        isLoading={isLoading}
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