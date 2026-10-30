<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

// ---- 统一请求封装：带 token，401 清登录态 ----
async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) }
  headers.Authorization = `Bearer ${localStorage.getItem('token')}`
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }
  const resp = await fetch(path, { ...options, headers })
  if (resp.status === 401) {
    localStorage.clear()
    window.location.href = '/login'
    throw new Error('登录已过期')
  }
  const data = await resp.json().catch(() => ({}))
  if (!resp.ok) throw new Error(data.detail || `请求失败（${resp.status}）`)
  return data
}

// ---- 上传文档 ----
const file = ref(null)
const uploadMsg = ref('')
const uploading = ref(false)

function onFileChange(e) {
  file.value = e.target.files[0] || null
  uploadMsg.value = ''
}

async function handleUpload() {
  uploadMsg.value = ''
  if (!file.value) {
    uploadMsg.value = '请先选择文件'
    return
  }
  uploading.value = true
  try {
    const fd = new FormData()
    fd.append('file', file.value)
    await api('/api/admin/documents', { method: 'POST', body: fd })
    uploadMsg.value = `「${file.value.name}」入库成功`
    file.value = null
    await loadDocs()
  } catch (e) {
    uploadMsg.value = `入库失败：${e.message}`
  } finally {
    uploading.value = false
  }
}

// ---- 历史入库文档 ----
const docs = ref([])
const loadingDocs = ref(false)

async function loadDocs() {
  loadingDocs.value = true
  try {
    docs.value = await api('/api/admin/documents')
  } catch (e) {
    alert(`加载文档列表失败：${e.message}`)
  } finally {
    loadingDocs.value = false
  }
}

async function handleDelete(doc) {
  if (!confirm(`确定删除文档「${doc.doc_name}」？删除后该文档将无法被检索。`)) return
  try {
    await api(`/api/admin/documents/${encodeURIComponent(doc.doc_id)}`, { method: 'DELETE' })
    await loadDocs()
  } catch (e) {
    alert(`删除失败：${e.message}`)
  }
}

// ---- 热门工单 Top5 ----
const tickets = ref([])
const loadingTickets = ref(false)

async function loadTickets() {
  loadingTickets.value = true
  try {
    tickets.value = await api('/api/admin/hot-tickets')
  } catch (e) {
    alert(`加载热门工单失败：${e.message}`)
  } finally {
    loadingTickets.value = false
  }
}

// ---- 转人工工单（管理端闭环） ----
const handoffTickets = ref([])
const loadingHandoff = ref(false)
const handoffFilter = ref('待处理')   // 待处理 / 处理中 / 已关闭 / 全部
const selectedTicket = ref(null)
const resolutionDraft = ref('')

async function loadHandoffTickets() {
  loadingHandoff.value = true
  try {
    const params = new URLSearchParams()
    if (handoffFilter.value && handoffFilter.value !== '全部') params.set('status', handoffFilter.value)
    params.set('source', 'human_request')
    handoffTickets.value = await api(`/api/admin/tickets?${params.toString()}`)
  } catch (e) {
    alert(`加载工单失败：${e.message}`)
  } finally {
    loadingHandoff.value = false
  }
}

function openTicket(t) {
  selectedTicket.value = t
  resolutionDraft.value = t.resolution || ''
}

async function claimTicket(t) {
  try {
    await api(`/api/admin/tickets/${t.ticket_id}/claim`, { method: 'POST' })
    await loadHandoffTickets()
    selectedTicket.value = handoffTickets.value.find(x => x.ticket_id === t.ticket_id) || null
  } catch (e) {
    alert(`认领失败：${e.message}`)
  }
}

async function closeTicket(t) {
  if (!resolutionDraft.value.trim()) {
    alert('请先填写处理结果')
    return
  }
  try {
    await api(`/api/admin/tickets/${t.ticket_id}`, {
      method: 'PUT',
      body: JSON.stringify({ resolution: resolutionDraft.value, status: '已关闭' }),
    })
    await loadHandoffTickets()
    selectedTicket.value = null
  } catch (e) {
    alert(`结单失败：${e.message}`)
  }
}

onMounted(() => {
  loadDocs()
  loadTickets()
  loadHandoffTickets()
})
</script>

<template>
  <div class="admin-wrap">
    <header class="admin-top">
      <span class="admin-title">管理后台</span>
      <div class="admin-actions">
        <button class="btn-ghost" @click="router.push('/chat')">返回聊天</button>
        <button class="btn-ghost" @click="router.push('/login')">退出登录</button>
      </div>
    </header>

    <!-- 上传文档 -->
    <section class="admin-card">
      <h2>上传文档入库</h2>
      <div class="upload-row">
        <input class="file-input" type="file" accept=".txt,.md,.doc,.docx,.pdf" @change="onFileChange" />
        <button class="btn" :disabled="uploading" @click="handleUpload">
          {{ uploading ? '入库中…' : '上传入库' }}
        </button>
      </div>
      <p class="hint">{{ uploadMsg || '支持 txt / md / doc / docx / pdf，同文件名重复上传会覆盖旧版本' }}</p>
    </section>

    <!-- 历史入库文档 -->
    <section class="admin-card">
      <h2>历史入库文档（{{ docs.length }}）</h2>
      <table class="admin-table">
        <thead>
          <tr>
            <th>文档名</th>
            <th>块数</th>
            <th>类别</th>
            <th>入库时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="d in docs" :key="d.doc_id">
            <td>
              <span class="doc-name">{{ d.doc_name }}</span>
              <span class="doc-id">{{ d.doc_id }}</span>
            </td>
            <td>{{ d.chunk_count }}</td>
            <td>{{ d.categories.join('、') || '—' }}</td>
            <td>{{ d.created_at }}</td>
            <td>
              <button class="btn-danger" @click="handleDelete(d)">删除</button>
            </td>
          </tr>
          <tr v-if="!loadingDocs && docs.length === 0">
            <td colspan="5" class="empty">暂无入库文档</td>
          </tr>
        </tbody>
      </table>
    </section>

    <!-- 热门工单 -->
    <section class="admin-card">
      <h2>热门工单 Top 5（按累计出现次数）</h2>
      <table class="admin-table">
        <thead>
          <tr>
            <th>问题</th>
            <th>类别</th>
            <th>次数</th>
            <th>最近时间</th>
            <th>处理结果</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(t, i) in tickets" :key="i">
            <td>{{ t.question }}</td>
            <td>{{ t.category }}</td>
            <td>{{ t.occurrences }}</td>
            <td>{{ t.last_occurred_at || '—' }}</td>
            <td>{{ t.resolution || '—' }}</td>
          </tr>
          <tr v-if="!loadingTickets && tickets.length === 0">
            <td colspan="5" class="empty">暂无工单记录</td>
          </tr>
        </tbody>
      </table>
    </section>

    <!-- 转人工工单 -->
    <section class="admin-card">
      <h2>转人工工单（{{ handoffTickets.length }}）</h2>
      <div class="upload-row" style="margin-bottom:12px">
        <button v-for="s in ['待处理','处理中','已关闭','全部']" :key="s"
                class="btn-ghost" :class="{active: handoffFilter===s}"
                @click="handoffFilter=s; loadHandoffTickets()">{{ s }}</button>
        <button class="btn-ghost" @click="loadHandoffTickets()">刷新</button>
      </div>
      <table class="admin-table">
        <thead>
          <tr>
            <th>工单号</th><th>问题</th><th>用户</th><th>状态</th><th>负责人</th><th>创建时间</th><th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="t in handoffTickets" :key="t.ticket_id" @click="openTicket(t)" style="cursor:pointer">
            <td><span class="doc-id">{{ t.ticket_id }}</span></td>
            <td>{{ t.question }}</td>
            <td>{{ t.user_id }}</td>
            <td><span :class="['badge', 'badge-' + t.status]">{{ t.status }}</span></td>
            <td>{{ t.assignee || '—' }}</td>
            <td>{{ t.created_at }}</td>
            <td>
              <button v-if="t.status==='待处理'" class="btn-danger" @click.stop="claimTicket(t)">认领</button>
            </td>
          </tr>
          <tr v-if="!loadingHandoff && handoffTickets.length === 0">
            <td colspan="7" class="empty">暂无转人工工单</td>
          </tr>
        </tbody>
      </table>

      <!-- 详情 / 处理 -->
      <div v-if="selectedTicket" class="handoff-detail">
        <h3>工单详情 {{ selectedTicket.ticket_id }}</h3>
        <p><b>问题：</b>{{ selectedTicket.question }}</p>
        <p><b>处理过程：</b>{{ selectedTicket.process }}</p>
        <p><b>当前状态：</b>{{ selectedTicket.status }} ｜ <b>负责人：</b>{{ selectedTicket.assignee || '—' }}</p>
        <label>处理结果</label>
        <textarea v-model="resolutionDraft" rows="3"></textarea>
        <div class="upload-row" style="margin-top:8px">
          <button v-if="selectedTicket.status==='待处理'" class="btn" style="width:auto"
                  @click="claimTicket(selectedTicket)">认领并接手</button>
          <button class="btn" style="width:auto" @click="closeTicket(selectedTicket)">填写结果并结单</button>
          <button class="btn-ghost" @click="selectedTicket=null">关闭</button>
        </div>
      </div>
    </section>
  </div>
</template>
