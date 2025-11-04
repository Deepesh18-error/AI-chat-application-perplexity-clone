import React, { useRef, useEffect } from 'react';

import { BsTrash } from 'react-icons/bs';
import { BsPlusLg } from 'react-icons/bs';
import { TfiLayoutSidebarLeft } from "react-icons/tfi";

function Sidebar({ isOpen, onNewChat, onSessionSelect, currentSessionId, sessions, error, onSessionDelete , toggleSidebar, }) {

  const chatListRef = useRef(null);

  useEffect(() => {
    // This effect will run every time the 'isOpen' prop changes.
    // We only want to act when the sidebar OPENS.
    if (isOpen && chatListRef.current) {
      // If the sidebar is open and our ref is attached to the element,
      // force its scroll position to the very top.
      chatListRef.current.scrollTop = 0;
    }
  }, [isOpen]); // The dependency array ensures this runs only when 'isOpen' changes.


  const sidebarClassName = `sidebar ${isOpen ? 'open' : 'closed'}`;

  const handleDeleteClick = (e, sessionId) => {
    e.stopPropagation(); 
    onSessionDelete(sessionId);
  };

  return (
     <div className={sidebarClassName}>
      <div className="sidebar-content">
        <div className="sidebar-header">
          <h1>MEV</h1>
          <button onClick={toggleSidebar} className="sidebar-toggle-btn-internal">
            <TfiLayoutSidebarLeft />
          </button>
        </div>
        
        <button className="new-chat-btn" onClick={onNewChat}>
          <BsPlusLg />
          New Chat
        </button>

        {/*  STEP 4: Attach the ref to the scrollable div  */}
        <div ref={chatListRef} className="chat-history-list">
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
                  <BsTrash />
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