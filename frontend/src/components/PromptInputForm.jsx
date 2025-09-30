// In frontend/src/components/PromptInputForm.jsx

import React from 'react';

/**
 * A "dumb" presentational component for the prompt input form.
 * It receives all its data and behavior via props from a parent component.
 *
 * @param {object} props - The component's properties.
 * @param {string} props.prompt - The current value of the input field.
 * @param {function} props.setPrompt - The function to call when the input value changes.
 * @param {function} props.handleSubmit - The function to call when the form is submitted.
 * @param {boolean} props.isLoading - A flag to disable the form during submission.
 * @returns {JSX.Element} The rendered form element.
 */
function PromptInputForm({ prompt, setPrompt, handleSubmit, isLoading }) {
  return (
    <form onSubmit={handleSubmit} className="prompt-form">
      <input
        type="text"
        className="prompt-input"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Ask anything..."
        disabled={isLoading}
        aria-label="Ask anything"
        autoFocus
      />
      <button type="submit" className="prompt-button" disabled={isLoading}>
        {isLoading ? 'Thinking...' : 'Ask'}
      </button>
    </form>
  );
}

export default PromptInputForm;