import { axiosInstance } from '@/api/lightrag'

export type Params = Record<string, string | number | undefined | null>

const clean = (params: Params = {}) =>
  Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ''))

const get = async (path: string, params?: Params) =>
  (await axiosInstance.get(`/admin/insights${path}`, { params: clean(params) })).data

export const insightsApi = {
  status: () => get('/status'),
  overview: (p: Params) => get('/overview', p),
  purpose: (id: string, p: Params) => get(`/purposes/${id}`, p),
  workspaces: (p: Params) => get('/workspaces', p),
  workspaceOptions: (p: Params) => get('/workspace-options', p),
  workspace: (id: string, p: Params) => get(`/workspaces/${encodeURIComponent(id)}`, p),
  breakdown: (p: Params) => get('/breakdown', p),
  calls: (p: Params) => get('/calls', p),
  day: (day: string, p: Params) => get(`/days/${day}`, p),
  anomalies: (p: Params) => get('/anomalies', p),
  setAnomaly: async (id: string, state: string) =>
    (await axiosInstance.post(`/admin/insights/anomalies/${encodeURIComponent(id)}`, { state })).data,
  work: (p: Params) => get('/work', p),
  forecast: (p: Params) => get('/forecast', p),
  settings: () => get('/settings'),
  saveSettings: async (patch: Record<string, any>) =>
    (await axiosInstance.put('/admin/insights/settings', patch)).data,
  backfill: async () => (await axiosInstance.post('/admin/insights/backfill')).data
}

/** Download through axios so the admin's auth header goes along. */
export const downloadFile = async (path: string, filename: string, params?: Params) => {
  const response = await axiosInstance.get(`/admin/insights${path}`, {
    params: clean(params),
    responseType: 'blob'
  })
  const url = URL.createObjectURL(response.data)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
