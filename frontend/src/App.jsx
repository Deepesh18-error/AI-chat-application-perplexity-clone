// src/App.jsx - RESTRUCTURED FOR NEW LAYOUT

import { useState, useEffect, useRef } from 'react';
import { ThemeProvider } from '@thesysai/genui-sdk';
import ResponseContainer from './components/ResponseContainer';
import WelcomeScreen from './components/WelcomeScreen';
import Sidebar from './components/Sidebar'; // <-- STEP 1: Import the new component
import './index.css';
import { v4 as uuidv4 } from 'uuid';
import { BsMicFill } from 'react-icons/bs';

function App() {
  const [prompt, setPrompt] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false); 

  const [sessions, setSessions] = useState([]);
  const [sessionsError, setSessionsError] = useState(null);

  const [forceWebSearch, setForceWebSearch] = useState(false);

  const [isSpeechRecognitionSupported, setIsSpeechRecognitionSupported] = useState(false);

  const [isListening, setIsListening] = useState(false);
  const speechRecognitionRef = useRef(null);


    useEffect(() => {
    // --- START OF NEW SPEECH RECOGNITION LOGIC ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      setIsSpeechRecognitionSupported(true);
      
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        for (let i = 0; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript + ' ';
          } else {
            interimTranscript += transcript;
          }
        }
        // Update the prompt state with the live transcription
        setPrompt(finalTranscript + interimTranscript);
      };

      recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false); // Turn off listening state on error
      };

      recognition.onend = () => {
        setIsListening(false); // Ensure listening is off when recognition ends
      };

      // Store the configured instance in our ref
      speechRecognitionRef.current = recognition;
      
    } else {
      setIsSpeechRecognitionSupported(false);
    }
    // --- END OF NEW SPEECH RECOGNITION LOGIC ---
  }, []);

  useEffect(() => {
    const fetchSessions = async () => {
      setSessionsError(null); 
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL}sessions/`);
        if (!response.ok) throw new Error(`Network response was not ok (${response.status})`);
        const data = await response.json();
        setSessions(data);
      } catch (error) {
        console.error("Failed to fetch sessions:", error);
        setSessionsError("Could not load chats."); 
      }
    };
        if (isSidebarOpen) {
        fetchSessions();
    }
    }, [isSidebarOpen, sessionId]);
   const toggleSidebar = () => {
    setIsSidebarOpen(prev => !prev);
  };

  const handleMicClick = () => {
    if (!speechRecognitionRef.current) {
      return; // Do nothing if speech recognition is not supported/initialized
    }

    if (isListening) {
      // If already listening, stop it
      speechRecognitionRef.current.stop();
      setIsListening(false);
    } else {
      // If not listening, start it
      speechRecognitionRef.current.start();
      setIsListening(true);
    }
  };

    const handleNewChat = () => {
    // Reset the chat history to an empty array
    setChatHistory([]);
    // Clear the session ID so a new one is generated on the next message
    setSessionId(null);
    // A nice UX touch: close the sidebar after starting a new chat
    setIsSidebarOpen(false);
  };

    const handleLoadSession = async (sessionIdToLoad) => {
    // Get the sessionId from the component's state for comparison
    const currentSessionIdFromState = sessionId;

    // --- LOGIC CHECK 1: Don't do anything if no ID is provided ---
    if (!sessionIdToLoad) {
      console.warn("[SESSION] handleLoadSession called with no ID. Aborting.");
      return;
    }

    // --- LOGIC CHECK 2: Don't reload if the requested session is already loaded ---
    if (sessionIdToLoad === currentSessionIdFromState) {
      console.log(`[SESSION] Session ${sessionIdToLoad} is already active. Closing sidebar.`);
      setIsSidebarOpen(false); // Just close the sidebar as a UX improvement
      return;
    }

    // --- If checks pass, proceed with loading ---
    console.log(`[SESSION] Starting to load new session: ${sessionIdToLoad}`);
    setIsLoading(true);
    setChatHistory([]); // Clear the old chat
    setIsSidebarOpen(false); // Close the sidebar

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}sessions/${sessionIdToLoad}/`);
      if (!response.ok) {
        throw new Error(`Failed to fetch session history: ${response.statusText}`);
      }
      const data = await response.json();

      // --- CRITICAL STATE UPDATES ---
      setChatHistory(data);         // Load the new history
      setSessionId(sessionIdToLoad); // Set the new session as active
      
      console.log(`[SESSION] Successfully loaded ${data.length} turns for session ${sessionIdToLoad}.`);
    } catch (error) {
      console.error("Error loading session:", error);
      setChatHistory([]); // Clear history on error
      setSessionId(null);   // Reset session ID on error
    } finally {
      setIsLoading(false);
    }
};




  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    // ... (Your entire handleSubmit logic remains UNCHANGED)
    // No need to copy it here, it stays exactly the same.
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
    images: [],
  };
    setChatHistory(prev => [...prev, newResponseState]);
    console.log("  [STATE] Initial response object added to chat history.");

    try {
        const requestPayload = {
        prompt: currentPrompt,
        session_id: currentSessionId,
        turn_number: chatHistory.length + 1,
        context_package: context_package,
        force_web_search: forceWebSearch,
      };
      console.log("  [DEBUG] Payload being sent to backend:", requestPayload);

      const response = await fetch(`${import.meta.env.VITE_API_URL}generate/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload),

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
                
                case 'images': {
                  try {
                    const eventData = JSON.parse(reconstructedData);
                    updatedState.images = eventData.images;
                    console.log("  [STATE] Received and stored image data.", eventData.images);
                  } catch (e) {
                    console.error("  [STATE] Failed to parse images JSON:", e, reconstructedData);
                  }
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

  const handleDeleteSession = async (sessionIdToDelete) => {
    console.log(`[SESSION] Attempting to delete session: ${sessionIdToDelete}`);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}sessions/${sessionIdToDelete}/`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`Failed to delete session. Status: ${response.status}`);
      }

      console.log(`[SESSION] Successfully deleted on backend. Updating UI.`);
      
      // 1. Update the sidebar list immediately for instant feedback
      setSessions(prevSessions => prevSessions.filter(s => s.session_id !== sessionIdToDelete));

      // 2. If the deleted chat is the one currently open, reset the main view
      if (sessionIdToDelete === sessionId) {
        console.log(`[SESSION] Active session was deleted. Resetting chat view.`);
        setChatHistory([]);
        setSessionId(null);
      }
    } catch (error) {
      console.error("Error deleting session:", error);
      // Optionally, set an error state to show a notification to the user
    }
  };

  return (
    <ThemeProvider>
      {/* STEP 2: The entire structure is replaced with the new layout */}
      <div className="app-layout">
        <Sidebar 
          isOpen={isSidebarOpen} 
          onNewChat={handleNewChat}
          onSessionSelect={handleLoadSession}
          currentSessionId={sessionId}
          sessions={sessions} // Pass the session list
          error={sessionsError}   // Pass any errors
          onSessionDelete={handleDeleteSession} // Pass the delete handler
        />
        <div className={`main-content ${isSidebarOpen ? 'sidebar-is-open' : ''}`}>
          <button onClick={toggleSidebar} className="sidebar-toggle-btn">
            ☰
          </button>
          <div className="chat-area"> {/* Renamed from response-area */}
            {chatHistory.length === 0 ? (
              <WelcomeScreen 
                onExampleClick={handleExampleClick}
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
            ) : (
              chatHistory.map((chat) => (
                <ResponseContainer key={chat.key} response={chat} />
              ))
            )}
          </div>

          <div className="prompt-section"> {/* Wrapper div for form */}
            <form onSubmit={handleSubmit} className="prompt-form">

              <button
                type="button" // Important: type="button" prevents form submission
                onClick={() => setForceWebSearch(prev => !prev)}
                className={`web-search-toggle ${forceWebSearch ? 'active' : ''}`}
                title="Force Web Search"
              >
                🌐
              </button>

              {isSpeechRecognitionSupported && (
                <button
                  type="button"
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
        </div>
      </div>
    </ThemeProvider>
  );
}

export default App;