import axios from 'axios';

const apiClient = axios.create({
  baseURL: '',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

export default apiClient;
