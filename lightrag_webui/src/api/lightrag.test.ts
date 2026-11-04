import { afterEach, beforeAll, describe, expect, test } from 'bun:test'

type DocumentsRequest = {
  status_filter?: 'pending' | 'processing' | 'preprocessed' | 'processed' | 'failed' | null
  page: number
  page_size: number
  sort_field: 'created_at' | 'updated_at' | 'id' | 'file_path'
  sort_direction: 'asc' | 'desc'
}

type LightragApiModule = typeof import('./lightrag')

const storageMock = () => {
  const data = new Map<string, string>()

  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value)
    },
    removeItem: (key: string) => {
      data.delete(key)
    },
    clear: () => {
      data.clear()
    }
  }
}

let apiModule: LightragApiModule

beforeAll(async () => {
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock(),
    configurable: true
  })
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: storageMock(),
    configurable: true
  })

  apiModule = await import('./lightrag')
})

afterEach(() => {
  apiModule.__resetPaginatedDocumentRequestsForTests()
})

describe('getDocumentsPaginated', () => {
  test('issues a fresh request after aborting a timed-out in-flight request', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    const resolvers: Array<(value: any) => void> = []

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolvers.push(resolve)
        controller.signal.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          { once: true }
        )
      })
    })

    const firstRequest = apiModule.getDocumentsPaginated(request)
    const secondRequest = apiModule.getDocumentsPaginated(request)

    expect(callCount).toBe(1)

    apiModule.abortDocumentsPaginated(request)
    const [firstResult, secondResult] = await Promise.allSettled([
      firstRequest,
      secondRequest
    ])
    expect(firstResult.status).toBe('rejected')
    expect(secondResult.status).toBe('rejected')

    const thirdRequest = apiModule.getDocumentsPaginated(request)
    expect(callCount).toBe(2)

    resolvers[1]({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(thirdRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })

  test('times out hanging requests and allows a fresh retry', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    const resolvers: Array<(value: any) => void> = []

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolvers.push(resolve)
        controller.signal.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          { once: true }
        )
      })
    })

    await expect(
      apiModule.getDocumentsPaginatedWithTimeout(request, 1)
    ).rejects.toThrow('Document fetch timeout')

    expect(callCount).toBe(1)

    const retryRequest = apiModule.getDocumentsPaginated(request)
    expect(callCount).toBe(2)

    resolvers[1]({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(retryRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })

  test('does not abort a shared request when only one timeout subscriber expires', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    let resolveSharedRequest: ((value: any) => void) | undefined
    let abortCount = 0

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolveSharedRequest = resolve
        controller.signal.addEventListener(
          'abort',
          () => {
            abortCount += 1
            reject(new DOMException('Aborted', 'AbortError'))
          },
          { once: true }
        )
      })
    })

    const shortTimeoutRequest = apiModule.getDocumentsPaginatedWithTimeout(request, 1)
    const longTimeoutRequest = apiModule.getDocumentsPaginatedWithTimeout(request, 100)

    await expect(shortTimeoutRequest).rejects.toThrow('Document fetch timeout')

    expect(callCount).toBe(1)
    expect(abortCount).toBe(0)

    resolveSharedRequest?.({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(longTimeoutRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })
})

describe('toApiError', () => {
  test('keeps the message format and attaches status, detail and request id', async () => {
    const { errorMessage } = await import('@/lib/utils')
    const err = apiModule.toApiError(
      {
        status: 500,
        statusText: 'Internal Server Error',
        data: { detail: 'Internal Server Error', request_id: 'body-id' },
        headers: { 'x-request-id': '3c9e41d0' }
      },
      '/documents/upload'
    )
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toBe('500 Internal Server Error\n{"detail":"Internal Server Error","request_id":"body-id"}\n/documents/upload')
    expect(err.status).toBe(500)
    expect(err.detail).toBe('Internal Server Error')
    // The response header wins over the body copy.
    expect(err.requestId).toBe('3c9e41d0')
    // Toasts built with errorMessage() name the request so support can find the log line.
    expect(errorMessage(err)).toBe(`${err.message}\nRef: 3c9e41d0`)
  })

  test('reads the header through AxiosHeaders-style get() and falls back to the body id', () => {
    const viaGet = apiModule.toApiError({ status: 403, statusText: 'Forbidden', data: { detail: 'Admin only' }, headers: { get: (k: string) => (k === 'x-request-id' ? 'hdr' : undefined) } })
    expect(viaGet.requestId).toBe('hdr')
    expect(viaGet.detail).toBe('Admin only')
    const viaBody = apiModule.toApiError({ status: 500, data: { detail: 'x', request_id: 'body-only' }, headers: {} })
    expect(viaBody.requestId).toBe('body-only')
    const none = apiModule.toApiError({ status: 502, statusText: 'Bad Gateway', data: '<html>proxy</html>' })
    expect(none.requestId).toBeNull()
    expect(none.detail).toBeNull()
  })

  test('errorMessage leaves errors without a request id unchanged', async () => {
    const { errorMessage } = await import('@/lib/utils')
    expect(errorMessage(new Error('Authentication required'))).toBe('Authentication required')
    expect(errorMessage('plain')).toBe('plain')
  })
})
