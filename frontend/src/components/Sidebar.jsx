// src/components/Sidebar.jsx

import React, { useState, useEffect } from 'react';

function Sidebar({ isOpen, onNewChat, onSessionSelect, currentSessionId }) {
  const [sessions, setSessions] = useState([]);
  // --- ADD THIS LINE: State to hold any error messages ---
  const [error, setError] = useState(null); 
  const sidebarClassName = `sidebar ${isOpen ? 'open' : 'closed'}`;

  useEffect(() => {
    const fetchSessions = async () => {
      // --- ADD THIS LINE: Reset error state on new fetch attempt ---
      setError(null); 
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL}sessions/`);
        if (!response.ok) {
          throw new Error(`Network response was not ok (${response.status})`);
        }
        const data = await response.json();
        setSessions(data);
      } catch (error) {
        console.error("Failed to fetch sessions:", error);
        // --- ADD THIS LINE: Set the error state for the UI ---
        setError("Could not load chats."); 
      }
    };
    
    if (isOpen) {
        fetchSessions();
    }
  }, [isOpen, currentSessionId]);

  return (
    <div className={sidebarClassName}>
      <div className="sidebar-content">
        <button className="new-chat-btn" onClick={onNewChat}>
          ➕ New Chat
        </button>

        <div className="chat-history-list">
          <h3 className="history-title">Chats</h3>
          
          {/* --- REPLACE THE EXISTING MAPPING LOGIC WITH THIS CONDITIONAL BLOCK --- */}
          {error ? (
            <p className="sidebar-error-text">{error}</p>
          ) : sessions.length > 0 ? (
            sessions.map((session) => (
              <button
                key={session.session_id}
                className={`chat-history-item ${
                  session.session_id === currentSessionId ? 'active' : ''
                }`}
                onClick={() => onSessionSelect(session.session_id)}
                title={session.title}
              >
                {session.title}
              </button>
            ))
          ) : (
            <p className="no-history-text">No chat history yet.</p>
          )}
          {/* --- END OF REPLACEMENT --- */}
        </div>
      </div>
    </div>
  );
}

export default Sidebar;