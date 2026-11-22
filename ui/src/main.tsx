import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { MutationCache, QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App'
import { ApiError } from './api'
import { toast } from './toast'

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
  // every action that fails says so once, in a toast; pages show their own loading errors in place
  mutationCache: new MutationCache({
    onError: (error) => toast(error.message, { details: error instanceof ApiError ? error.details : undefined }),
  }),
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
