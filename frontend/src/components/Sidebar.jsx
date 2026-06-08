import { useEffect, useRef } from 'react';
import { BsChatLeftText, BsClockHistory, BsPlusLg, BsTrash } from 'react-icons/bs';
import { TfiLayoutSidebarLeft } from 'react-icons/tfi';

function Sidebar({
  isOpen,
  onNewChat,
  onSessionSelect,
  currentSessionId,
  sessions,
  isLoading,
  error,
  onSessionDelete,
  toggleSidebar,
}) {
  const chatListRef = useRef(null);

  useEffect(() => {
    if (isOpen && chatListRef.current) {
      chatListRef.current.scrollTop = 0;
    }
  }, [isOpen]);

  const sidebarClassName = `sidebar ${isOpen ? 'open' : 'closed'}`;

  const handleDeleteClick = (event, sessionId) => {
    event.stopPropagation();
    onSessionDelete(sessionId);
  };

  const handleSessionKeyDown = (event, sessionId) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    onSessionSelect(sessionId);
  };

  return (
    <div className={sidebarClassName}>
      <div className="sidebar-content">
        <div className="sidebar-header">
          <h1>ARGON</h1>
          <button type="button" onClick={toggleSidebar} className="sidebar-toggle-btn-internal" title="Close chat history">
            <TfiLayoutSidebarLeft />
          </button>
        </div>

        <button type="button" className="new-chat-btn" onClick={onNewChat}>
          <BsPlusLg />
          New Chat
        </button>

        <div className="history-section-header">
          <div>
            <span className="history-eyebrow">Workspace</span>
            <h3 className="history-title">Chat history</h3>
          </div>
          <span className="history-count">{sessions.length}</span>
        </div>

        <div ref={chatListRef} className="chat-history-list">
          {isLoading ? (
            <div className="history-loading-state" aria-label="Loading chat history">
              <span />
              <span />
              <span />
            </div>
          ) : error ? (
            <p className="sidebar-error-text">{error}</p>
          ) : sessions.length > 0 ? (
            sessions.map((session) => (
              <div
                key={session.session_id}
                className={`chat-history-item ${session.session_id === currentSessionId ? 'active' : ''}`}
                onClick={() => onSessionSelect(session.session_id)}
                onKeyDown={(event) => handleSessionKeyDown(event, session.session_id)}
                role="button"
                tabIndex={0}
                title={session.title}
              >
                <span className="history-item-icon">
                  <BsChatLeftText />
                </span>
                <span className="history-item-content">
                  <span className="history-item-title">{session.title}</span>
                  <span className="history-item-meta">Saved conversation</span>
                </span>
                <button
                  type="button"
                  className="delete-button"
                  onClick={(event) => handleDeleteClick(event, session.session_id)}
                  title="Delete chat"
                >
                  <BsTrash />
                </button>
              </div>
            ))
          ) : (
            <div className="no-history-state">
              <BsClockHistory />
              <p>No chat history yet.</p>
              <span>Your saved conversations will appear here.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Sidebar;
