// src/App.jsx - ENHANCED LOGGING

import { useState } from 'react';
import { ThemeProvider } from '@thesysai/genui-sdk';
import ResponseContainer from './components/ResponseContainer';
import WelcomeScreen from './components/WelcomeScreen';
import './index.css';
import { v4 as uuidv4 } from 'uuid';

function App() {
  const [prompt, setPrompt] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    console.group(`🚀 [SUBMIT] New Request Started: "${prompt}"`);
    setIsLoading(true);
    const currentPrompt = prompt;
    setPrompt('');

    let currentSessionId = sessionId;
    if (!currentSessionId) {
      const newSessionId = uuidv4();
      setSessionId(newSessionId);
      currentSessionId = newSessionId;
      console.log(`  [SESSION] New session started with ID: ${newSessionId}`);
    } else {
      console.log(`  [SESSION] Continuing session with ID: ${currentSessionId}`);
    }

  const context_package = {
    current_query: currentPrompt,
    previous_turns: chatHistory
      .map(turn => ({
        query: turn.prompt,
        summary: turn.summary,
        entities: turn.entities,
      })),
  };
    console.log("  [CONTEXT] Assembled context package:", context_package);


  const newResponseState = {
    key: Date.now(),
    prompt: currentPrompt,
    steps: [],
    sources: [],
    auiSpec: null,
    error: null,
    isLoading: true,
    summary: null,
    entities: [],
  };
    setChatHistory(prev => [...prev, newResponseState]);
    console.log("  [STATE] Initial response object added to chat history.");

    try {
      const response = await fetch(import.meta.env.VITE_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: currentPrompt,
          session_id: currentSessionId,
          turn_number: chatHistory.length + 1, // The new turn number
          context_package: context_package,
        }),

      });
      console.log("  [NETWORK] Initial response received from backend. Status:", response.status);

      if (!response.ok || !response.body) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      console.log("  [STREAM] Reader created. Starting stream processing loop.");

      const processStream = async () => {
        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            console.log("  [STREAM] Stream is DONE.");
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const messages = buffer.split('\n\n');
          buffer = messages.pop() || '';
          for (const message of messages) {
            if (message.trim() === '') continue;

            console.groupCollapsed("  [STREAM] Processing SSE Message Block");
            console.log("Raw Message Block:", `"${message.replace(/\n/g, '\\n')}"`);

            let eventType = 'message'; // Default event type
            const dataBuffer = [];

            const lines = message.split('\n');
            for (const line of lines) {
              if (line.startsWith('event: ')) {
                eventType = line.substring(7).trim();
              } else if (line.startsWith('data: ')) {
                dataBuffer.push(line.substring(6)); // Push the content after "data: "
              }
            }

            // Reconstruct the full data payload from all data lines
            const reconstructedData = dataBuffer.join('\n');
            
            if (!reconstructedData) {
                console.groupEnd();
                continue;
            }

            console.log("Event Type:", eventType);
            console.log("Reconstructed Data:", reconstructedData);

            setChatHistory(prev => prev.map((chat, index) => {
              if (index !== prev.length - 1) return chat;
              let updatedState = { ...chat };

              switch (eventType) {
                case 'steps':
                case 'sources':
                case 'error': {
                  const eventData = JSON.parse(reconstructedData);
                  if (eventType === 'steps') updatedState.steps = [...updatedState.steps, eventData.message];
                  if (eventType === 'sources') updatedState.sources = eventData.sources;
                  if (eventType === 'error') updatedState.error = eventData.message;
                  break;
                }
                
                case 'aui_dsl': {
                  // The reconstructedData IS the full, raw C1 DSL string.
                  updatedState.auiSpec = reconstructedData;
                  break;
                }
                
                 case 'turn_metadata': {
                  try {
                      const metadata = JSON.parse(reconstructedData);
                      updatedState.summary = metadata.summary;
                      updatedState.entities = metadata.entities;
                      console.log("  [STATE] Received and stored turn metadata. Turn complete.", metadata);

                      // --- CRITICAL FIX ---
                      // Since metadata is the TRUE final step, we now unlock the main UI here.
                      setIsLoading(false); 
                      console.log("  [STATE] UI unlocked after receiving metadata.");
                      // --- END OF FIX ---

                    } catch (e) {
                      console.error("  [STATE] Failed to parse turn_metadata JSON:", e, reconstructedData);
                      setIsLoading(false); // Also unlock on error
                    }
                    break;
                  }

                case 'finished':
                  break;
                  
                default:
                  console.warn("-> Received unknown event type:", eventType);
              }

              return updatedState;
            }));

            console.groupEnd();
          }
        }
      };
      await processStream();
    } catch (error) {
      console.error('❌ [FETCH ERROR] An error occurred during the fetch process:', error);
      setChatHistory(prev => prev.map((chat, i) => i === prev.length - 1 ? { ...chat, error: error.message, isLoading: false } : chat));
      setIsLoading(false);
    } finally {
      
      console.log("[FINALLY] isLoading set to false, UI unlocked.");
      console.groupEnd();
    }
  };

  const handleExampleClick = (examplePrompt) => {
    setPrompt(examplePrompt);
  };

  return (
    <ThemeProvider>
      <div className="app-container">
        <div className="response-area">
          {chatHistory.length === 0 ? (
            <WelcomeScreen onExampleClick={handleExampleClick} />
          ) : (
            chatHistory.map((chat) => (
              <ResponseContainer key={chat.key} response={chat} />
            ))
          )}
        </div>

        <form onSubmit={handleSubmit} className="prompt-form">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ask me anything..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? '...' : 'Ask'}
          </button>
        </form>
      </div>
    </ThemeProvider>
  );
}

export default App;