import { Component, ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { Empty } from './primitives'

function ViewFailed() {
  const { t } = useTranslation()
  return <Empty>{t('admin.viewFailed')}</Empty>
}

/**
 * Keeps one view's failure from blanking the whole app. A new `resetKey`
 * tries again. `name` labels the console error ("Token Insights view
 * failed: …").
 */
export class ViewBoundary extends Component<{ resetKey?: string; name?: string; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error: Error) {
    console.error(`${this.props.name ?? 'Admin'} view failed:`, error)
  }

  componentDidUpdate(previous: { resetKey?: string }) {
    if (this.state.failed && previous.resetKey !== this.props.resetKey) this.setState({ failed: false })
  }

  render() {
    return this.state.failed ? <ViewFailed /> : this.props.children
  }
}

export default ViewBoundary
