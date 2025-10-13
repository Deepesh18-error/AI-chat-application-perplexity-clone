import React from 'react';

function WelcomeScreen({ onExampleClick }) {
  return (
    <div className="welcome-screen">
      <h1>Perplexity Clone</h1>
      <p>Ask anything. Get direct answers or comprehensive results from the web.</p>
      <div className="example-prompts">
        <button onClick={() => onExampleClick("What is Bernoulli's principle?")}>
          → What is Bernoulli's principle?
        </button>
        <button onClick={() => onExampleClick("Create a short story about a brave knight")}>
          → Create a short story...
        </button>
        <button onClick={() => onExampleClick("Who is the current CEO of OpenAI?")}>
          → Who is the current CEO of OpenAI?
        </button>
      </div>
    </div>
  );
}

export default WelcomeScreen;