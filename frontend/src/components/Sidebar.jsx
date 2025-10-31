// src/components/Sidebar.jsx - FINAL VERSION

import React from 'react';
import { BsTrash } from 'react-icons/bs';
import { BsPlusLg } from 'react-icons/bs';
import { TfiLayoutSidebarLeft } from "react-icons/tfi";



// ... (function signature remains the same)
function Sidebar({ isOpen, onNewChat, onSessionSelect, currentSessionId, sessions, error, onSessionDelete , toggleSidebar, }) {
  // ... (sidebarClassName and handleDeleteClick function are unchanged)
  const sidebarClassName = `sidebar ${isOpen ? 'open' : 'closed'}`;

  const handleDeleteClick = (e, sessionId) => {
    e.stopPropagation(); 
    onSessionDelete(sessionId);
  };

  return (
     <div className={sidebarClassName}>
      <div className="sidebar-content">
        {/* --- ADD THIS NEW HEADER SECTION --- */}
        <div className="sidebar-header">
          <h1>MEV</h1>
          <button onClick={toggleSidebar} className="sidebar-toggle-btn-internal">
            <TfiLayoutSidebarLeft />
          </button>
        </div>
        {/* --- END OF NEW HEADER --- */}

        {/* --- UPDATE THE NEW CHAT BUTTON --- */}
        <button className="new-chat-btn" onClick={onNewChat}>
          <BsPlusLg   /> {/* <-- Use the icon component */}
          New Chat
        </button>

        <div className="chat-history-list">
          <h3 className="history-title">Chats</h3>
          
          {error ? (
            <p className="sidebar-error-text">{error}</p>
          ) : sessions.length > 0 ? (
            sessions.map((session) => (
              <div
                key={session.session_id}
                className={`chat-history-item ${session.session_id === currentSessionId ? 'active' : ''}`}
                onClick={() => onSessionSelect(session.session_id)}
                title={session.title}
              >
                <span className="history-item-title">{session.title}</span>
                <button 
                  className="delete-button"
                  onClick={(e) => handleDeleteClick(e, session.session_id)}
                  title="Delete chat"
                >
                  <BsTrash /> {/* <--- 2. USE THE ICON COMPONENT */}
                </button>
              </div>
            ))
          ) : (
            <p className="no-history-text">No chat history yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default Sidebar;