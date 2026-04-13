import { useState, useEffect, useRef } from 'react';
import { ThemeProvider } from '@thesysai/genui-sdk';
import { themePresets } from '@crayonai/react-ui';
import ResponseContainer from './components/ResponseContainer';
import WelcomeScreen from './components/WelcomeScreen';
import Sidebar from './components/Sidebar'; 
import './index.css';
import { v4 as uuidv4 } from 'uuid';
import { HiGlobe, HiMicrophone } from 'react-icons/hi';
import TextareaAutosize from 'react-textarea-autosize';
import { TfiLayoutSidebarLeft } from "react-icons/tfi"; 
import { flushSync } from 'react-dom';


function App() {
  const [prompt, setPrompt] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(
  () => localStorage.getItem('argon_last_session_id') || null
);
  const [isRestoring, setIsRestoring] = useState(
  () => !!localStorage.getItem('argon_last_session_id')
);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false); 

  const [sessions, setSessions] = useState([]);
  const [sessionsError, setSessionsError] = useState(null);

  const [forceWebSearch, setForceWebSearch] = useState(false);

  const [isSpeechRecognitionSupported, setIsSpeechRecognitionSupported] = useState(false);

  const [isListening, setIsListening] = useState(false);
  const speechRecognitionRef = useRef(null);

  useEffect(() => {
    if (sessionId) {
      localStorage.setItem('argon_last_session_id', sessionId);
    }
    // NOTE: We do NOT remove on null here.
    // Removal is done explicitly in handleNewChat and handleDeleteSession only.
  }, [sessionId]);

  // ── ON MOUNT: restore last session from localStorage ──
useEffect(() => {
  const savedSessionId = localStorage.getItem('argon_last_session_id');
  if (!savedSessionId) {
    setIsRestoring(false);
    return;
  }

  console.log(`[RESTORE] Found saved session: ${savedSessionId}. Loading...`);

  fetch(`${import.meta.env.VITE_API_URL}sessions/${savedSessionId}/`)
    .then(res => {
      if (!res.ok) throw new Error('Session not found');
      return res.json();
    })
    .then(data => {
      if (data && data.length > 0) {
        setChatHistory(data);
        // sessionId already set by lazy initializer, no need to set again
        console.log(`[RESTORE] Successfully restored ${data.length} turns.`);
      } else {
        localStorage.removeItem('argon_last_session_id');
      }
    })
    .catch(err => {
      console.warn('[RESTORE] Could not restore session:', err);
      localStorage.removeItem('argon_last_session_id');
    })
    .finally(() => {
      setIsRestoring(false);
    });
}, []);


    useEffect(() => {

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

        setPrompt(finalTranscript + interimTranscript);
      };

      recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      speechRecognitionRef.current = recognition;
      
    } else {
      setIsSpeechRecognitionSupported(false);
    }

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


  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };


  const handleMicClick = () => {
    if (!speechRecognitionRef.current) {
      return;
    }

    if (isListening) {
      speechRecognitionRef.current.stop();
      setIsListening(false);
    } else {
      speechRecognitionRef.current.start();
      setIsListening(true);
    }
  };

  const handleNewChat = () => {
    setChatHistory([]);
    setSessionId(null);
    localStorage.removeItem('argon_last_session_id'); 
    setIsSidebarOpen(false);
  };

  const handleLoadSession = async (sessionIdToLoad) => {
    const currentSessionIdFromState = sessionId;

    if (!sessionIdToLoad) {
      console.warn("[SESSION] handleLoadSession called with no ID. Aborting.");
      return;
    }

    if (sessionIdToLoad === currentSessionIdFromState) {
      console.log(`[SESSION] Session ${sessionIdToLoad} is already active. Closing sidebar.`);
      setIsSidebarOpen(false);
      return;
    }

    console.log(`[SESSION] Starting to load new session: ${sessionIdToLoad}`);
    setIsLoading(true);
    setChatHistory([]);
    setIsSidebarOpen(false);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}sessions/${sessionIdToLoad}/`);
      if (!response.ok) {
        throw new Error(`Failed to fetch session history: ${response.statusText}`);
      }
      const data = await response.json();

      setChatHistory(data);
      setSessionId(sessionIdToLoad);
      
      console.log(`[SESSION] Successfully loaded ${data.length} turns for session ${sessionIdToLoad}.`);
    } catch (error) {
      console.error("Error loading session:", error);
      setChatHistory([]);
      setSessionId(null);
    } finally {
      setIsLoading(false);
    }
  };




  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    const uiClickTime = Date.now(); 
    console.log(`⏱ [UI] 'Enter' pressed at: ${new Date(uiClickTime).toISOString()}`);

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
      previous_turns: chatHistory.map(turn => ({
        query: turn.prompt,
        summary: turn.summary,
        entities: turn.entities,
      })),
    };
    console.log("  [CONTEXT] Assembled context package:", context_package);

    const newResponseState = {
      key: Date.now(),
      prompt: currentPrompt,
      progress: {
        path: null,
        currentStage: 'analyzing',
        queriesGenerated: [],
        sourcesFound: 0,
        sourcesBeingScraped: [],
        totalScraped: 0,
      },
      steps: [],
      sources: [],
      images: [],
      auiSpec: null,
      error: null,
      summary: null,
      entities: [],
      streamingMarkdown: '',
      isLoadedFromHistory: false,
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
        client_start_time: uiClickTime
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

            let eventType = 'message';
            const dataBuffer = [];

            const lines = message.split('\n');
            for (const line of lines) {
              if (line.startsWith('event: ')) {
                eventType = line.substring(7).trim();
              } else if (line.startsWith('data: ')) {
                dataBuffer.push(line.substring(6));
              }
            }

            const reconstructedData = dataBuffer.join('\n');
            
            if (!reconstructedData) {
              console.groupEnd();
              continue;
            }

            console.log("Event Type:", eventType);
            console.log("Reconstructed Data:", reconstructedData);

            // ── CASE 1: markdown_chunk ──
            // Handled separately with flushSync for live writing effect.
            // Each chunk paints to the DOM immediately.
            if (eventType === 'markdown_chunk') {
              const eventData = JSON.parse(reconstructedData);
              flushSync(() => {
                setChatHistory(prev => prev.map((chat, index) => {
                  if (index !== prev.length - 1) return chat;
                  return {
                    ...chat,
                    streamingMarkdown: (chat.streamingMarkdown || '') + eventData.chunk,
                    progress: { ...chat.progress, currentStage: 'synthesizing' }
                  };
                }));
              });
              console.groupEnd();
              continue;
            }
            // ── CASE 3: turn_metadata ──
            // Handled separately with flushSync so setIsLoading(false) fires
            // only AFTER auiSpec is already committed (from CASE 2 above).
            if (eventType === 'turn_metadata') {
              try {
                const metadata = JSON.parse(reconstructedData);
                flushSync(() => {
                  setChatHistory(prev => prev.map((chat, index) => {
                    if (index !== prev.length - 1) return chat;
                    return {
                      ...chat,
                      summary: metadata.summary,
                      entities: metadata.entities,
                      progress: { ...chat.progress, currentStage: 'complete' }
                    };
                  }));
                });
                setIsLoading(false);
                console.log("  [STATE] UI unlocked after receiving metadata.");
              } catch (e) {
                console.error("  [STATE] Failed to parse turn_metadata JSON:", e, reconstructedData);
                setIsLoading(false);
              }
              console.groupEnd();
              continue;
            }

            // ── ALL OTHER EVENTS ──
            // These are batched normally by React — no flushSync needed.
            setChatHistory(prev => prev.map((chat, index) => {
              if (index !== prev.length - 1) return chat;
              let updatedState = { ...chat };
              if (!updatedState.progress) {
                updatedState.progress = {};
              }

              switch (eventType) {

                case 'analysis_complete': {
                  const eventData = JSON.parse(reconstructedData);
                  updatedState.progress = { 
                    ...updatedState.progress, 
                    path: eventData.path,
                    currentStage: eventData.path === 'direct_answer' ? 'synthesizing' : 'searching'
                  };
                  break;
                }

                case 'query_generated': {
                  const eventData = JSON.parse(reconstructedData);
                  updatedState.progress = {
                    ...updatedState.progress,
                    queriesGenerated: [...updatedState.progress.queriesGenerated, eventData.query]
                  };
                  break;
                }

                case 'urls_complete': {
                  console.log("  [STATE] URL retrieval complete.");
                  break;
                }

                case 'scraping_complete': {
                  const eventData = JSON.parse(reconstructedData);
                  updatedState.progress = {
                    ...updatedState.progress,
                    totalScraped: eventData.total_scraped
                  };
                  break;
                }

                case 'synthesis_start': {
                  updatedState.progress = {
                    ...updatedState.progress,
                    currentStage: 'synthesizing'
                  };
                  break;
                }

                case 'source_retrieved': {
                  const eventData = JSON.parse(reconstructedData);
                  updatedState.progress = { 
                    ...updatedState.progress, 
                    currentStage: 'retrieving',
                    sourcesRetrieved: eventData.count
                  };
                  break;
                }

                case 'context_complete': {
                  console.log("  [STATE] Retrieval and Scraping complete.");
                  break;
                }
                
                case 'aui_dsl': {
                  updatedState.auiSpec = reconstructedData;
                  break;
                }

                case 'queries_complete': {
                  console.log("  [STATE] All queries generated.");
                  break;
                }

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
      
      setSessions(prevSessions => prevSessions.filter(s => s.session_id !== sessionIdToDelete));

      if (sessionIdToDelete === sessionId) {
        console.log(`[SESSION] Active session was deleted. Resetting chat view.`);
        setChatHistory([]);
        setSessionId(null);
        localStorage.removeItem('argon_last_session_id');

      }
    } catch (error) {
      console.error("Error deleting session:", error);
    }
  };

  return (
    <ThemeProvider
      theme={themePresets.carbon}
      darkTheme={themePresets.carbon}
      mode="dark"
    >
      {}
      <div className={`app-layout ${isSidebarOpen ? 'sidebar-open' : ''}`}>
        {chatHistory.length > 0 && (
          <Sidebar 
            isOpen={isSidebarOpen} 
            onNewChat={handleNewChat}
            onSessionSelect={handleLoadSession}
            toggleSidebar={toggleSidebar}
            currentSessionId={sessionId}
            sessions={sessions}
            error={sessionsError}
            onSessionDelete={handleDeleteSession}
          />
        )}
        <div className={`main-content ${isSidebarOpen ? 'sidebar-is-open' : ''}`}>
          {chatHistory.length > 0 && (
            <button onClick={toggleSidebar} className="sidebar-toggle-btn">
              <TfiLayoutSidebarLeft />
            </button>
          )}
      <div className="chat-area">
        {isRestoring ? (
          <div className="restore-loading">
            <div className="restore-spinner" />
            <p>Restoring your session...</p>
          </div>
        ) : chatHistory.length === 0 ? (
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
          chatHistory.map((chat, index) => ( 
            <div key={chat.key} className="turn-container">
              <ResponseContainer 
                response={chat} 
                isLastTurn={index === chatHistory.length - 1} 
              />
            </div>
          ))
        )}
      </div>

          <div className="prompt-section">
            <form onSubmit={handleSubmit} className="prompt-form">
              <TextareaAutosize
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Ask me anything..."
                disabled={isLoading}
                onKeyDown={handleKeyDown}
                rows={1}
                maxRows={8}
                className="prompt-textarea"
              />

              <div className="prompt-actions-bar">
                <div className="prompt-actions-left">
                  <button
                    type="button"
                    className={`action-btn web-search-toggle ${forceWebSearch ? 'active' : ''}`}
                    onClick={() => setForceWebSearch(!forceWebSearch)}
                    title="Force Web Search"
                  >
                    <HiGlobe />
                  </button>
                  <button
                    type="button"
                    className={`action-btn ${isListening ? 'active' : ''}`}
                    onClick={handleMicClick} 
                    title="Voice Input"
                  >
                    <HiMicrophone />
                  </button>
                </div>
                
                <div className="prompt-actions-right">
                  <button 
                    type="submit" 
                    className="submit-btn" 
                    disabled={isLoading || !prompt.trim()}
                  >
                    Ask
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
}

export default App;