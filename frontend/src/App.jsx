import { useState, useEffect, useRef } from 'react';
import { flushSync } from 'react-dom';
import { ThemeProvider } from '@thesysai/genui-sdk';
import { themePresets } from '@crayonai/react-ui';
import { v4 as uuidv4 } from 'uuid';
import { HiGlobe, HiMicrophone } from 'react-icons/hi';
import { TfiLayoutSidebarLeft } from 'react-icons/tfi';
import TextareaAutosize from 'react-textarea-autosize';

import AuthScreen from './components/AuthScreen';
import ResponseContainer from './components/ResponseContainer';
import WelcomeScreen from './components/WelcomeScreen';
import Sidebar from './components/Sidebar';
import {
  AuthError,
  authFetch,
  clearAuth,
  getMe,
  getStoredAuth,
  logout,
} from './services/authClient';
import './index.css';

function App() {
  const [auth, setAuth] = useState(() => getStoredAuth());
  const [isAuthChecking, setIsAuthChecking] = useState(true);
  const [prompt, setPrompt] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [isRestoring, setIsRestoring] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [isSessionsLoading, setIsSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState(null);
  const [forceWebSearch, setForceWebSearch] = useState(false);
  const [isSpeechRecognitionSupported, setIsSpeechRecognitionSupported] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const speechRecognitionRef = useRef(null);
  const currentUser = auth?.user || null;
  const lastSessionStorageKey = currentUser ? `argon_last_session_id_${currentUser.id}` : null;

  useEffect(() => {
    let isMounted = true;

    const validateAuth = async () => {
      if (!getStoredAuth()) {
        setIsAuthChecking(false);
        return;
      }

      try {
        const nextAuth = await getMe();
        if (isMounted) setAuth(nextAuth);
      } catch (error) {
        console.error('[AUTH] Session validation failed:', error);
        clearAuth();
        if (isMounted) setAuth(null);
      } finally {
        if (isMounted) setIsAuthChecking(false);
      }
    };

    validateAuth();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (sessionId && lastSessionStorageKey) {
      localStorage.setItem(lastSessionStorageKey, sessionId);
    }
  }, [lastSessionStorageKey, sessionId]);

  useEffect(() => {
    if (!currentUser || !lastSessionStorageKey) {
      setIsRestoring(false);
      return;
    }

    const savedSessionId = localStorage.getItem(lastSessionStorageKey);
    if (!savedSessionId) {
      setIsRestoring(false);
      return;
    }

    setIsRestoring(true);

    authFetch(`sessions/${savedSessionId}/`)
      .then((res) => {
        if (!res.ok) throw new Error('Session not found');
        return res.json();
      })
      .then((data) => {
        if (data?.length > 0) {
          setChatHistory(data);
          setSessionId(savedSessionId);
        } else {
          localStorage.removeItem(lastSessionStorageKey);
        }
      })
      .catch((err) => {
        console.warn('[RESTORE] Could not restore session:', err);
        localStorage.removeItem(lastSessionStorageKey);
        if (err instanceof AuthError) {
          clearAuth();
          setAuth(null);
        }
      })
      .finally(() => {
        setIsRestoring(false);
      });
  }, [currentUser, lastSessionStorageKey]);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setIsSpeechRecognitionSupported(false);
      return;
    }

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
          finalTranscript += `${transcript} `;
        } else {
          interimTranscript += transcript;
        }
      }

      setPrompt(finalTranscript + interimTranscript);
    };

    recognition.onerror = (event) => {
      console.error('[SPEECH] Recognition error:', event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    speechRecognitionRef.current = recognition;

    return () => {
      recognition.stop();
      speechRecognitionRef.current = null;
    };
  }, []);

  useEffect(() => {
    const fetchSessions = async () => {
      setSessionsError(null);
      setIsSessionsLoading(true);
      try {
        const response = await authFetch('sessions/');
        if (!response.ok) throw new Error(`Network response was not ok (${response.status})`);
        const data = await response.json();
        setSessions(data);
      } catch (error) {
        console.error('[SESSIONS] Failed to fetch sessions:', error);
        if (error instanceof AuthError) {
          clearAuth();
          setAuth(null);
        }
        setSessionsError('Could not load chats.');
      } finally {
        setIsSessionsLoading(false);
      }
    };

    if (isSidebarOpen && currentUser) {
      fetchSessions();
    }
  }, [currentUser, isSidebarOpen, sessionId]);

  const toggleSidebar = () => {
    setIsSidebarOpen((prev) => !prev);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleMicClick = () => {
    if (!speechRecognitionRef.current) return;

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
    if (lastSessionStorageKey) {
      localStorage.removeItem(lastSessionStorageKey);
    }
    setIsSidebarOpen(false);
  };

  const handleLoadSession = async (sessionIdToLoad) => {
    if (!sessionIdToLoad) return;

    if (sessionIdToLoad === sessionId) {
      setIsSidebarOpen(false);
      return;
    }

    setIsLoading(true);
    setChatHistory([]);
    setIsSidebarOpen(false);

    try {
      const response = await authFetch(`sessions/${sessionIdToLoad}/`);
      if (!response.ok) {
        throw new Error(`Failed to fetch session history: ${response.statusText}`);
      }

      const data = await response.json();
      setChatHistory(data);
      setSessionId(sessionIdToLoad);
    } catch (error) {
      console.error('[SESSIONS] Error loading session:', error);
      if (error instanceof AuthError) {
        clearAuth();
        setAuth(null);
      }
      setChatHistory([]);
      setSessionId(null);
    } finally {
      setIsLoading(false);
    }
  };

  const updateLastChat = (updater) => {
    setChatHistory((prev) => prev.map((chat, index) => {
      if (index !== prev.length - 1) return chat;
      return updater(chat);
    }));
  };

  const parseJsonEvent = (data) => JSON.parse(data);

  const handleStreamEvent = (eventType, reconstructedData) => {
    if (eventType === 'markdown_chunk') {
      const eventData = parseJsonEvent(reconstructedData);
      flushSync(() => {
        updateLastChat((chat) => ({
          ...chat,
          streamingMarkdown: (chat.streamingMarkdown || '') + eventData.chunk,
          progress: { ...chat.progress, currentStage: 'synthesizing' },
        }));
      });
      return;
    }

    if (eventType === 'turn_metadata') {
      try {
        const metadata = parseJsonEvent(reconstructedData);
        flushSync(() => {
          updateLastChat((chat) => ({
            ...chat,
            summary: metadata.summary,
            entities: metadata.entities,
            progress: { ...chat.progress, currentStage: 'complete' },
          }));
        });
      } catch (error) {
        console.error('[STREAM] Failed to parse turn metadata:', error);
      } finally {
        setIsLoading(false);
      }
      return;
    }

    if (eventType === 'provider_warning') {
      const eventData = parseJsonEvent(reconstructedData);
      updateLastChat((chat) => ({
        ...chat,
        providerWarnings: [...(chat.providerWarnings || []), eventData.message],
      }));
      return;
    }

    updateLastChat((chat) => {
      const updatedState = {
        ...chat,
        progress: chat.progress || {},
      };

      switch (eventType) {
        case 'analysis_complete': {
          const eventData = parseJsonEvent(reconstructedData);
          updatedState.progress = {
            ...updatedState.progress,
            path: eventData.path,
            currentStage: eventData.path === 'direct_answer' ? 'synthesizing' : 'searching',
          };
          break;
        }

        case 'synthesis_start': {
          updatedState.progress = {
            ...updatedState.progress,
            currentStage: 'synthesizing',
          };
          break;
        }

        case 'aui_dsl': {
          updatedState.auiSpec = reconstructedData;
          break;
        }

        case 'steps': {
          const eventData = parseJsonEvent(reconstructedData);
          updatedState.steps = [...updatedState.steps, eventData.message];
          break;
        }

        case 'sources': {
          const eventData = parseJsonEvent(reconstructedData);
          updatedState.sources = eventData.sources;
          updatedState.progress = {
            ...updatedState.progress,
            currentStage: 'retrieving',
            sourcesRetrieved: eventData.sources?.length || 0,
          };
          break;
        }

        case 'images': {
          const eventData = parseJsonEvent(reconstructedData);
          updatedState.images = eventData.images || [];
          break;
        }

        case 'error': {
          const eventData = parseJsonEvent(reconstructedData);
          updatedState.error = eventData.message;
          break;
        }

        case 'finished':
          break;

        default:
          console.warn('[STREAM] Unknown event type:', eventType);
      }

      return updatedState;
    });
  };

  const processStream = async (reader) => {
    const decoder = new TextDecoder();
    let buffer = '';

    let isReading = true;

    while (isReading) {
      const { value, done } = await reader.read();
      if (done) {
        isReading = false;
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const messages = buffer.split('\n\n');
      buffer = messages.pop() || '';

      for (const message of messages) {
        if (!message.trim()) continue;

        let eventType = 'message';
        const dataBuffer = [];

        for (const line of message.split('\n')) {
          if (line.startsWith('event: ')) {
            eventType = line.substring(7).trim();
          } else if (line.startsWith('data: ')) {
            dataBuffer.push(line.substring(6));
          }
        }

        const reconstructedData = dataBuffer.join('\n');
        if (reconstructedData) {
          handleStreamEvent(eventType, reconstructedData);
        }
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    const uiClickTime = Date.now();
    setIsLoading(true);

    const currentPrompt = prompt;
    setPrompt('');

    let currentSessionId = sessionId;
    if (!currentSessionId) {
      currentSessionId = uuidv4();
      setSessionId(currentSessionId);
    }

    const context_package = {
      current_query: currentPrompt,
      previous_turns: chatHistory.map((turn) => ({
        query: turn.prompt,
        summary: turn.summary,
        entities: turn.entities,
      })),
    };

    const newResponseState = {
      key: Date.now(),
      prompt: currentPrompt,
      progress: {
        path: null,
        currentStage: 'analyzing',
        queriesGenerated: [],
        sourcesRetrieved: 0,
        totalScraped: 0,
      },
      steps: [],
      sources: [],
      images: [],
      auiSpec: null,
      error: null,
      providerWarnings: [],
      summary: null,
      entities: [],
      streamingMarkdown: '',
      isLoadedFromHistory: false,
    };

    setChatHistory((prev) => [...prev, newResponseState]);

    try {
      const requestPayload = {
        prompt: currentPrompt,
        session_id: currentSessionId,
        context_package,
        force_web_search: forceWebSearch,
        client_start_time: uiClickTime,
      };

      const response = await authFetch('generate/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      await processStream(response.body.getReader());
    } catch (error) {
      console.error('[FETCH] Request failed:', error);
      if (error instanceof AuthError) {
        clearAuth();
        setAuth(null);
      }
      updateLastChat((chat) => ({ ...chat, error: error.message, isLoading: false }));
      setIsLoading(false);
    }
  };

  const handleExampleClick = (examplePrompt) => {
    setPrompt(examplePrompt);
  };

  const handleDeleteSession = async (sessionIdToDelete) => {
    try {
      const response = await authFetch(`sessions/${sessionIdToDelete}/`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`Failed to delete session. Status: ${response.status}`);
      }

      setSessions((prevSessions) => prevSessions.filter((s) => s.session_id !== sessionIdToDelete));

      if (sessionIdToDelete === sessionId) {
        setChatHistory([]);
        setSessionId(null);
        if (lastSessionStorageKey) {
          localStorage.removeItem(lastSessionStorageKey);
        }
      }
    } catch (error) {
      console.error('[SESSIONS] Error deleting session:', error);
      if (error instanceof AuthError) {
        clearAuth();
        setAuth(null);
      }
    }
  };

  const resetWorkspace = () => {
    setPrompt('');
    setChatHistory([]);
    setSessionId(null);
    setSessions([]);
    setIsSessionsLoading(false);
    setSessionsError(null);
    setIsSidebarOpen(false);
    setIsLoading(false);
    setIsRestoring(false);
  };

  const handleAuthSuccess = (nextAuth) => {
    setAuth(nextAuth);
    resetWorkspace();
  };

  const handleLogout = async () => {
    await logout();
    setAuth(null);
    resetWorkspace();
  };

  if (isAuthChecking) {
    return (
      <div className="auth-screen">
        <div className="auth-panel">
          <h1>ARGON</h1>
          <p>Checking your session...</p>
        </div>
      </div>
    );
  }

  if (!currentUser) {
    return <AuthScreen onAuthSuccess={handleAuthSuccess} />;
  }

  return (
    <ThemeProvider
      theme={themePresets.carbon}
      darkTheme={themePresets.carbon}
      mode="dark"
    >
      <div className={`app-layout ${isSidebarOpen ? 'sidebar-open' : ''}`}>
        <Sidebar
          isOpen={isSidebarOpen}
          onNewChat={handleNewChat}
          onSessionSelect={handleLoadSession}
          toggleSidebar={toggleSidebar}
          currentSessionId={sessionId}
          sessions={sessions}
          isLoading={isSessionsLoading}
          error={sessionsError}
          onSessionDelete={handleDeleteSession}
        />
        <div className={`main-content ${isSidebarOpen ? 'sidebar-is-open' : ''}`}>
          <div className="auth-user-bar">
            <span>{currentUser.email}</span>
            <button type="button" onClick={handleLogout}>Logout</button>
          </div>

          <button
            type="button"
            onClick={toggleSidebar}
            className="sidebar-toggle-btn"
            title="Open chat history"
            aria-label="Open chat history"
            aria-expanded={isSidebarOpen}
          >
            <TfiLayoutSidebarLeft />
          </button>

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
                user={currentUser}
              />
            ) : (
              chatHistory.map((chat) => (
                <div key={chat.key} className="turn-container">
                  <ResponseContainer response={chat} />
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
