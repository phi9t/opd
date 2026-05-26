export interface TokenSample {
  prompt: string
  tokens: string[]
  kl: number[]
}

export interface StepMetrics {
  step: number
  gen_ms: number
  teacher_ms: number
  train_ms: number
  sync_ms: number
  sync_bytes: number
  num_tokens: number
  mean_response_length: number
  mean_kl: number
  loss: number
  grad_norm: number
  loss_mode: string
  teacher_signal: string
  samples?: TokenSample[]
}

export interface RunBundle {
  run_id: string
  tier: string
  runtime: string
  device: string
  loss_mode: string
  teacher_signal: string
  models: {
    student_hidden_size: number
    teacher_hidden_size: number
    vocab_size: number
  }
  steps: StepMetrics[]
  glossary: Record<string, string>
}

export interface RunIndexEntry {
  id: string
  path: string
  tier: string
}
