const AUTH_STORAGE_KEY = 'argon_auth';

export const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api/').replace(/\/?$/, '/');

export const apiUrl = (path) => `${API_BASE_URL}${path.replace(/^\//, '')}`;

export class AuthError extends Error {
  constructor(message = 'Authentication required') {
    super(message);
    this.name = 'AuthError';
  }
}

export const getStoredAuth = () => {
  try {
    const rawAuth = localStorage.getItem(AUTH_STORAGE_KEY);
    return rawAuth ? JSON.parse(rawAuth) : null;
  } catch {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    return null;
  }
};

export const saveAuth = (authPayload) => {
  const currentAuth = getStoredAuth() || {};
  const nextAuth = {
    user: authPayload.user || currentAuth.user,
    accessToken: authPayload.access_token || authPayload.accessToken || currentAuth.accessToken,
    refreshToken: authPayload.refresh_token || authPayload.refreshToken || currentAuth.refreshToken,
  };

  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(nextAuth));
  return nextAuth;
};

export const clearAuth = () => {
  localStorage.removeItem(AUTH_STORAGE_KEY);
};

const parseErrorMessage = async (response) => {
  try {
    const data = await response.json();
    return data.error || data.message || `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
};

const publicJsonRequest = async (path, body) => {
  const response = await fetch(apiUrl(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return response.json();
};

export const login = async ({ email, password }) => {
  const data = await publicJsonRequest('auth/login/', { email, password });
  return saveAuth(data);
};

export const register = async ({ name, email, password }) => {
  const data = await publicJsonRequest('auth/register/', { name, email, password });
  return saveAuth(data);
};

export const refreshAuth = async () => {
  const currentAuth = getStoredAuth();
  if (!currentAuth?.refreshToken) {
    throw new AuthError();
  }

  const response = await fetch(apiUrl('auth/refresh/'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: currentAuth.refreshToken }),
  });

  if (!response.ok) {
    clearAuth();
    throw new AuthError(await parseErrorMessage(response));
  }

  const data = await response.json();
  return saveAuth({ ...data, user: currentAuth.user });
};

export const authFetch = async (path, options = {}, retry = true) => {
  const currentAuth = getStoredAuth();
  if (!currentAuth?.accessToken) {
    throw new AuthError();
  }

  const headers = new Headers(options.headers || {});
  headers.set('Authorization', `Bearer ${currentAuth.accessToken}`);

  const response = await fetch(apiUrl(path), {
    ...options,
    headers,
  });

  if (response.status === 401 && retry && currentAuth.refreshToken) {
    await refreshAuth();
    return authFetch(path, options, false);
  }

  if (response.status === 401) {
    clearAuth();
    throw new AuthError();
  }

  return response;
};

export const authJson = async (path, options = {}) => {
  const response = await authFetch(path, options);
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return response.json();
};

export const getMe = async () => {
  const data = await authJson('auth/me/');
  const currentAuth = getStoredAuth();
  return saveAuth({ ...currentAuth, user: data.user });
};

export const logout = async () => {
  try {
    await authFetch('auth/logout/', { method: 'POST' });
  } finally {
    clearAuth();
  }
};
