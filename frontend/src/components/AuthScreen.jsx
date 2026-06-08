import { useState } from 'react';
import { FiLock, FiMail, FiUser } from 'react-icons/fi';

import { login, register } from '../services/authClient';

function AuthScreen({ onAuthSuccess }) {
  const [mode, setMode] = useState('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const isRegistering = mode === 'register';

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setIsSubmitting(true);

    try {
      const auth = isRegistering
        ? await register({ name, email, password })
        : await login({ email, password });
      onAuthSuccess(auth);
    } catch (authError) {
      setError(authError.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="auth-screen">
      <div className="auth-circuit auth-circuit-left-top">
        <span className="auth-node-card" />
        <svg className="auth-circuit-path" viewBox="0 0 620 260" preserveAspectRatio="none">
          <path id="authPathLeftTop" d="M112 36 H430 V126 L620 236" />
          <circle className="auth-node-light" r="4">
            <animateMotion dur="26s" repeatCount="indefinite" keyPoints="0;1;0" keyTimes="0;0.5;1" calcMode="spline" keySplines="0.42 0 0.58 1;0.42 0 0.58 1">
              <mpath href="#authPathLeftTop" />
            </animateMotion>
          </circle>
        </svg>
      </div>
      <div className="auth-circuit auth-circuit-right-top">
        <span className="auth-node-card" />
        <svg className="auth-circuit-path" viewBox="0 0 620 260" preserveAspectRatio="none">
          <path id="authPathRightTop" d="M508 36 H220 V126 L0 236" />
          <circle className="auth-node-light" r="4">
            <animateMotion dur="28s" repeatCount="indefinite" keyPoints="0;1;0" keyTimes="0;0.5;1" calcMode="spline" keySplines="0.42 0 0.58 1;0.42 0 0.58 1">
              <mpath href="#authPathRightTop" />
            </animateMotion>
          </circle>
        </svg>
      </div>
      <div className="auth-circuit auth-circuit-left-bottom">
        <span className="auth-node-card" />
        <svg className="auth-circuit-path" viewBox="0 0 620 260" preserveAspectRatio="none">
          <path id="authPathLeftBottom" d="M112 224 H360 L430 164 V112 L620 20" />
          <circle className="auth-node-light" r="4">
            <animateMotion dur="30s" repeatCount="indefinite" keyPoints="0;1;0" keyTimes="0;0.5;1" calcMode="spline" keySplines="0.42 0 0.58 1;0.42 0 0.58 1">
              <mpath href="#authPathLeftBottom" />
            </animateMotion>
          </circle>
        </svg>
      </div>
      <div className="auth-circuit auth-circuit-right-bottom">
        <span className="auth-node-card" />
        <svg className="auth-circuit-path" viewBox="0 0 620 260" preserveAspectRatio="none">
          <path id="authPathRightBottom" d="M508 224 H280 L220 164 V112 L0 20" />
          <circle className="auth-node-light" r="4">
            <animateMotion dur="32s" repeatCount="indefinite" keyPoints="0;1;0" keyTimes="0;0.5;1" calcMode="spline" keySplines="0.42 0 0.58 1;0.42 0 0.58 1">
              <mpath href="#authPathRightBottom" />
            </animateMotion>
          </circle>
        </svg>
      </div>

      <form className="auth-panel" onSubmit={handleSubmit}>
        <div className="auth-brand-row">
          <span className="auth-dot-grid" />
          <div className="auth-logo">ARGON</div>
          <span className="auth-dot-grid" />
        </div>

        <div className="auth-copy">
          <h1>{isRegistering ? 'Create Workspace' : 'Welcome Back'}</h1>
          <p>
            {isRegistering ? 'Already have an account?' : "Don't have an account yet?"}
            <button
              type="button"
              onClick={() => {
                setMode(isRegistering ? 'login' : 'register');
                setError('');
              }}
            >
              {isRegistering ? 'Login' : 'Sign up'}
            </button>
          </p>
        </div>

        {isRegistering && (
          <label className="auth-field">
            <FiUser />
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="workspace name"
              autoComplete="name"
            />
          </label>
        )}

        <label className="auth-field">
          <FiMail />
          <input
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="email address"
            type="email"
            autoComplete="email"
            required
          />
        </label>

        <label className="auth-field">
          <FiLock />
          <input
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="password"
            type="password"
            autoComplete={isRegistering ? 'new-password' : 'current-password'}
            minLength={8}
            required
          />
        </label>

        {error && <p className="auth-error">{error}</p>}

        <button className="auth-primary" type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Please wait...' : isRegistering ? 'Create account' : 'Login'}
        </button>
      </form>
    </div>
  );
}

export default AuthScreen;
