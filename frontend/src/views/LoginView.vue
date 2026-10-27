<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { login, register } from '../api/auth'

const router = useRouter()
const tab = ref('login')
const errorMsg = ref('')

const loginForm = ref({ username: '', password: '' })
const regForm = ref({ username: '', password: '', confirm: '' })

function switchTab(t) {
  tab.value = t
  errorMsg.value = ''   // 切换 tab 时清空错误提示
}

function saveSession(data) {
  localStorage.setItem('token', data.token)
  localStorage.setItem('user_id', data.user_id)
  localStorage.setItem('username', data.username)
  localStorage.setItem('role', data.role || 'user')
}

async function handleLogin() {
  errorMsg.value = ''
  if (!loginForm.value.username.trim() || !loginForm.value.password) {
    errorMsg.value = '请输入用户名和密码'
    return
  }
  try {
    const data = await login(loginForm.value.username.trim(), loginForm.value.password)
    saveSession(data)
    router.push('/chat')
  } catch (e) {
    errorMsg.value = e.response?.data?.detail || '登录失败，请稍后重试'
  }
}

async function handleRegister() {
  errorMsg.value = ''
  const { username, password, confirm } = regForm.value
  if (username.trim().length < 2) {
    errorMsg.value = '用户名至少 2 个字符'
    return
  }
  if (password.length < 6) {
    errorMsg.value = '密码至少 6 位'
    return
  }
  if (password !== confirm) {
    errorMsg.value = '两次输入的密码不一致'
    return
  }
  try {
    const data = await register(username.trim(), password)
    saveSession(data)
    router.push('/chat')
  } catch (e) {
    errorMsg.value = e.response?.data?.detail || '注册失败，请稍后重试'
  }
}
</script>

<template>
  <div class="auth-wrap">
    <div class="card">
      <h1>智能客服</h1>

      <div class="tabs">
        <button class="tab-btn" :class="{ active: tab === 'login' }" @click="switchTab('login')">登录</button>
        <button class="tab-btn" :class="{ active: tab === 'register' }" @click="switchTab('register')">注册</button>
      </div>

      <!-- 统一错误提示区：与 tab 无关，永远显示在当前可见表单下方 -->
      <div class="error" v-if="errorMsg">{{ errorMsg }}</div>

      <form v-if="tab === 'login'" @submit.prevent="handleLogin">
        <div class="form-item">
          <label>用户名</label>
          <input v-model="loginForm.username" type="text" placeholder="请输入用户名" autocomplete="username" />
        </div>
        <div class="form-item">
          <label>密码</label>
          <input v-model="loginForm.password" type="password" placeholder="请输入密码" autocomplete="current-password" />
        </div>
        <button class="btn" type="submit">登 录</button>
      </form>

      <form v-else @submit.prevent="handleRegister">
        <div class="form-item">
          <label>用户名</label>
          <input v-model="regForm.username" type="text" placeholder="2-50 个字符" />
        </div>
        <div class="form-item">
          <label>密码</label>
          <input v-model="regForm.password" type="password" placeholder="至少 6 位" />
        </div>
        <div class="form-item">
          <label>确认密码</label>
          <input v-model="regForm.confirm" type="password" placeholder="再次输入密码" />
        </div>
        <button class="btn" type="submit">注 册</button>
      </form>
    </div>
  </div>
</template>
