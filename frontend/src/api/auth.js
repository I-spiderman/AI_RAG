import axios from 'axios'

// axios 实例：统一 baseURL（开发时由 Vite 代理到 FastAPI）
const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
})

// 请求拦截器：自动带 Authorization: Bearer <token>
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截器：401 说明登录态失效，清空并回登录页
api.interceptors.response.use(
  (resp) => resp.data,
  (err) => {
    if (err.response && err.response.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user_id')
      localStorage.removeItem('username')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  },
)

export function register(username, password) {
  return api.post('/auth/register', { username, password })
}

export function login(username, password) {
  return api.post('/auth/login', { username, password })
}

export function getMe() {
  return api.get('/auth/me')
}

export default api
