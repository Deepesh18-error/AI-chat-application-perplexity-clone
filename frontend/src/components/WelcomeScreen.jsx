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
  handleMicClick,
  user
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
        user={user}
      />
    </div>
  );
}

export default WelcomeScreen;
