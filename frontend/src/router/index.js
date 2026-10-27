import { createRouter, createWebHistory } from 'vue-router'
import LoginView from '../views/LoginView.vue'
import ChatView from '../views/ChatView.vue'
import AdminView from '../views/AdminView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/login' },
    { path: '/login', name: 'login', component: LoginView },
    { path: '/chat', name: 'chat', component: ChatView },
    { path: '/admin', name: 'admin', component: AdminView, meta: { admin: true } },
  ],
})

// 路由守卫：无 token 只能看登录页；已登录访问登录页则跳聊天页；管理员页仅 role=admin
router.beforeEach((to) => {
  const hasToken = !!localStorage.getItem('token')
  if (to.name !== 'login' && !hasToken) {
    return { name: 'login' }
  }
  if (to.name === 'login' && hasToken) {
    return { name: 'chat' }
  }
  if (to.meta?.admin && localStorage.getItem('role') !== 'admin') {
    return { name: 'chat' }
  }
})

export default router
