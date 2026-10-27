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

onMounted(() => {
  loadDocs()
  loadTickets()
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
  </div>
</template>
