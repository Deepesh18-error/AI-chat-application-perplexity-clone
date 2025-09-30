import React, { useState } from 'react';
import PromptInputForm from './components/PromptInputForm';
import { sendPrompt } from './services/apiClient';

// Styles remain here as they are global to this page.
const GlobalStyles = () => (
    <style>{`
      :root {
        --background-color: #f0f4f8;
        --container-bg: #ffffff;
        --text-color: #102a43;
        --primary-color: #3b82f6;
        --primary-hover: #2563eb;
        --border-color: #dcdfe6;
        --error-color: #d9534f;
        --font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      }
      *, *::before, *::after { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: var(--font-family);
        background-color: var(--background-color);
        color: var(--text-color);
        line-height: 1.6;
      }
      #root {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem;
        min-height: 100vh;
      }
      .chat-container {
        width: 100%;
        max-width: 800px;
        background-color: var(--container-bg);
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 1.5rem;
      }
      h1 {
        text-align: center;
        color: var(--primary-color);
        margin-bottom: 2rem;
      }
      .prompt-form { display: flex; gap: 0.5rem; }
      .prompt-input { flex-grow: 1; padding: 0.75rem 1rem; border: 1px solid var(--border-color); border-radius: 6px; font-size: 1rem; transition: border-color 0.2s; }
      .prompt-input:focus { outline: none; border-color: var(--primary-color); box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3); }
      .prompt-button { padding: 0.75rem 1.5rem; background-color: var(--primary-color); color: white; border: none; border-radius: 6px; font-size: 1rem; font-weight: 500; cursor: pointer; transition: background-color 0.2s; }
      .prompt-button:hover:not(:disabled) { background-color: var(--primary-hover); }
      .prompt-button:disabled { background-color: #93c5fd; cursor: not-allowed; }
      .error-message { color: var(--error-color); margin-top: 1rem; text-align: center; }
      .response-area { margin-top: 2rem; padding: 1rem; background-color: #f8fafc; border: 1px solid var(--border-color); border-radius: 6px; min-height: 100px; white-space: pre-wrap; font-family: 'Courier New', Courier, monospace; }
    `}</style>
);


function App() {
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!prompt || isLoading) return;

    setIsLoading(true);
    setError(null);
    setResponse('');

    try {
      // Logic is now delegated to the apiClient
      const data = await sendPrompt(prompt);
      console.log('✅ [Frontend] Response from backend:', data);
      setResponse(data.message);
    } catch (e) {
      console.error('🚨 [Frontend] There was an error making the request:', e);
      setError(e.message || 'Failed to connect to the backend. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <GlobalStyles />
      <div className="chat-container">
        <h1>Perplexity Clone</h1>
        <PromptInputForm
          prompt={prompt}
          setPrompt={setPrompt}
          handleSubmit={handleSubmit}
          isLoading={isLoading}
        />

        {error && <p className="error-message">{error}</p>}

        <div className="response-area">
          {response}
        </div>
      </div>
    </>
  );
}

export default App;