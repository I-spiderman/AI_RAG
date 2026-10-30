<script setup>
import { ref, nextTick, onMounted } from 'vue'

const username = ref(localStorage.getItem('username') || '')
const sessions = ref([])   // [{session_id, title, ...}]
const current = ref(null)  // 当前选中会话 {session_id, title}
const messages = ref([])   // {role: 'user'|'assistant', content, streaming, summary?}
const input = ref('')
const sending = ref(false)
const loading = ref(false)
const handoffInfo = ref(null)   // 转人工工单卡片：{ ticketId, message }

// 通用 fetch：自动带 Bearer，401 跳登录
async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${localStorage.getItem('token')}`,
      ...(options.headers || {}),
    },
    ...options,
  })
  if (resp.status === 401) {
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    window.location.href = '/login'
    throw new Error('登录已过期')
  }
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}))
    throw new Error(data.detail || `请求失败（${resp.status}）`)
  }
  return resp.json()
}

// 初始加载：拉会话列表，空则新建，非空则选中最新会话
onMounted(async () => {
  loading.value = true
  try {
    const data = await api('/api/sessions')
    sessions.value = data
    if (data.length > 0) {
      await selectSession(data[0].session_id, data[0].title)
    } else {
      await newChat()
    }
  } catch (e) {
    messages.value = [{ role: 'assistant', content: `会话加载失败：${e.message || e}` }]
  } finally {
    loading.value = false
  }
})

async function newChat() {
  if (sending.value) return
  const sid = crypto.randomUUID()
  try {
    await api('/api/sessions', { method: 'POST', body: JSON.stringify({ session_id: sid }) })
    const sess = { session_id: sid, title: '新对话' }
    sessions.value.unshift(sess)
    current.value = sess
    messages.value = [{ role: 'assistant', content: '您好，请问有什么可以帮您？' }]
    scrollBottom(true)
  } catch (e) {
    alert(`新建会话失败：${e.message || e}`)
  }
}

async function selectSession(sid, title) {
  if (sending.value) return
  // 直接引用列表项对象（同一引用），标题更新才能实时同步到侧栏
  const item = sessions.value.find((s) => s.session_id === sid)
  current.value = item || { session_id: sid, title }
  messages.value = []
  try {
    const data = await api(`/api/sessions/${sid}/messages`)
    const msgs = []
    if (data.summary) {
      msgs.push({ role: 'assistant', content: `【更早对话摘要】\n${data.summary}`, summary: true })
    }
    for (const [role, content] of data.messages) {
      msgs.push({ role, content })
    }
    if (msgs.length === 0) {
      msgs.push({ role: 'assistant', content: '您好，请问有什么可以帮您？' })
    }
    messages.value = msgs
    scrollBottom(true)
  } catch (e) {
    messages.value = [{ role: 'assistant', content: `历史加载失败：${e.message || e}` }]
  }
}

async function send() {
  const q = input.value.trim()
  if (!q || sending.value || !current.value) return
  input.value = ''
  messages.value.push({ role: 'user', content: q })
  // 关键：push 原始对象后必须取回【响应式代理】，否则打字机对 asst.content 的累加
  // 不会触发 Vue 重渲染——画面只会停在最后一次响应式变更时的部分内容（回答不完整/全弹出）
  messages.value.push({ role: 'assistant', content: '', streaming: true })
  const asst = messages.value[messages.value.length - 1]
  sending.value = true
  scrollBottom(true)   // 发新消息时强制回到底部
  try {
    await streamChat(q, asst)
    // 首条消息后更新本地标题（后端已自动落库）；双保险同步列表项
    if (current.value.title === '新对话') {
      current.value.title = q.slice(0, 20)
      const item = sessions.value.find((s) => s.session_id === current.value.session_id)
      if (item) item.title = current.value.title
    }
  } catch (e) {
    asst.content += e.message || '（回复中断，请重试）'
  } finally {
    asst.streaming = false
    sending.value = false
    scrollBottom()
  }
}

// SSE 流式接收：fetch + ReadableStream 解析 data 帧 + 打字机限速渲染
// 后端可能很快（短回答几秒内吐完所有帧），直接 append 会"一次性弹出"，
// 所以帧内容先入缓冲，由节拍器每 40ms 渲染 2 个字符，形成可感知的打字机效果。
async function streamChat(question, asst) {
  const resp = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
    body: JSON.stringify({ question, session_id: current.value.session_id }),
  })
  if (!resp.ok) {
    if (resp.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('username')
      window.location.href = '/login'
      throw new Error('登录已过期')
    }
    const data = await resp.json().catch(() => ({}))
    throw new Error(data.detail || `请求失败（${resp.status}）`)
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buf = ''
  let pending = ''          // 待渲染缓冲
  let ticker = null         // 打字机节拍器
  let closed = false        // 数据流是否已结束

  const startTicker = () => {
    if (ticker) return
    ticker = setInterval(() => {
      if (pending) {
        const take = Math.min(pending.length, 2)
        asst.content += pending.slice(0, take)
        pending = pending.slice(take)
        scrollBottom()
      } else if (closed) {
        // 缓冲已空且流已结束 → 停止节拍器，避免空转
        clearInterval(ticker)
        ticker = null
      }
    }, 40)
  }

  const finish = () => {
    // 收尾：把最后可能缺少 \n\n 的残留帧补进缓冲，交由打字机逐渐渲染
    if (buf) {
      const line = buf.split('\n').find((l) => l.startsWith('data: '))
      if (line) {
        try {
          const p = JSON.parse(line.slice(6))
          if (p.type === 'token') pending += p.content
          else if (p.type === 'error') pending += p.content || '（服务异常）'
          else if (p.type === 'handoff') handoffInfo.value = { ticketId: p.ticket_id, message: p.message }
        } catch (e) { /* ignore */ }
      }
      buf = ''
    }
    // 标记结束；不再一次性把 pending 塞入，让逐字效果延续到最后一字
    closed = true
  }

  try {
    startTicker()
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let idx
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        const line = frame.split('\n').find((l) => l.startsWith('data: '))
        if (!line) continue
        const payload = JSON.parse(line.slice(6))
        if (payload.type === 'token') {
          pending += payload.content
        } else if (payload.type === 'error') {
          pending += payload.content || '（服务异常）'
        } else if (payload.type === 'handoff') {
          // 转人工工单帧：保留工单号，流结束后渲染成工单卡片
          handoffInfo.value = { ticketId: payload.ticket_id, message: payload.message }
        }
        // status 帧（"正在思考…"）只作占位提示，由气泡的"正在输入…"承担，不写入回答正文
      }
    }
  } finally {
    finish()
    // 转人工工单卡片：回复文本渲染完成后追加到消息流，作为独立卡片气泡
    if (handoffInfo.value) {
      messages.value.push({
        role: 'assistant',
        handoff: true,
        ticketId: handoffInfo.value.ticketId,
        content: '',
      })
      scrollBottom(true)
      handoffInfo.value = null
    }
  }
}

// force=true 时强制滚到底（新消息/切换会话）；否则仅在用户本就停在底部附近时跟随
function scrollBottom(force = false) {
  nextTick(() => {
    const box = document.querySelector('.msg-list')
    if (!box) return
    const distance = box.scrollHeight - box.scrollTop - box.clientHeight
    if (force || distance < 120) box.scrollTop = box.scrollHeight
  })
}

function logout() {
  localStorage.removeItem('token')
  localStorage.removeItem('username')
  localStorage.removeItem('user_id')
  localStorage.removeItem('role')
  window.location.href = '/login'
}

const isAdmin = localStorage.getItem('role') === 'admin'
function goAdmin() {
  window.location.href = '/admin'
}
</script>

<template>
  <div class="chat-layout">
    <!-- 左侧：多会话列表 -->
    <aside class="sidebar">
      <div class="sidebar-head">
        <button class="new-chat-btn" @click="newChat">＋ 新对话</button>
      </div>
      <div class="session-list">
        <div
          v-for="s in sessions"
          :key="s.session_id"
          class="session-item"
          :class="{ active: current && current.session_id === s.session_id }"
          @click="selectSession(s.session_id, s.title)"
        >
          {{ s.title }}
        </div>
      </div>
    </aside>

    <!-- 右侧：聊天区 -->
    <div class="chat-wrap">
      <header class="chat-header">
        <span class="chat-title">{{ current ? current.title : '智能客服' }}</span>
        <span class="chat-actions">
          <button v-if="isAdmin" class="link-btn" @click="goAdmin">管理后台</button>
          <span class="chat-user">{{ username }}</span>
          <button class="link-btn" @click="logout">退出</button>
        </span>
      </header>

      <main class="msg-list">
        <div v-if="loading" class="msg-row assistant">
          <div class="bubble">加载中…</div>
        </div>
        <div v-for="(m, i) in messages" :key="i" class="msg-row" :class="[m.role, { handoff: m.handoff }]">
          <div
            class="bubble"
            :class="{ streaming: m.streaming, 'bubble-summary': m.summary }"
          >
            <span v-if="!m.content && m.streaming" class="typing">正在输入…</span>
            <template v-else-if="m.handoff">
              <div class="handoff-card">
                <div class="handoff-title">✅ 已为您转接人工客服</div>
                <div class="handoff-ticket">工单号：<b>{{ m.ticketId }}</b></div>
                <div class="handoff-tip">人工客服将尽快回复，您可凭工单号查询处理进度</div>
              </div>
            </template>
            <template v-else>{{ m.content }}</template>
          </div>
        </div>
      </main>

      <footer class="chat-input-bar">
        <div class="chat-input-inner">
          <input
            v-model="input"
            type="text"
            placeholder="请输入消息…（Enter 发送）"
            :disabled="sending"
            @keydown.enter="send"
          />
          <button class="btn send-btn" :disabled="sending" @click="send">
            {{ sending ? '发送中…' : '发送' }}
          </button>
        </div>
      </footer>
    </div>
  </div>
</template>
