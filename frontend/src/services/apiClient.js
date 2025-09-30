const API_BASE_URL = import.meta.env.VITE_API_URL;

/**
 * Sends a prompt to the backend API.
 * @param {string} prompt - The user's prompt to be sent.
 * @returns {Promise<object>} A promise that resolves to the JSON response from the backend.
 * @throws {Error} Throws an error if the network response is not ok.
 */

export const sendPrompt = async (prompt) => {
  // Construct the full URL for the API endpoint.
  const url = `${API_BASE_URL}/generate/`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ prompt }),
  });

  if (!response.ok) {
    // If the server responds with an error status (e.g., 400, 500),
    // we throw an error to be caught by the calling function.
    const errorBody = await response.json().catch(() => ({ error: 'Unknown error structure' }));
    throw new Error(`HTTP error! status: ${response.status}, message: ${errorBody.error || 'No error message provided'}`);
  }

  // If the response is successful, parse and return the JSON data.
  return response.json();
};