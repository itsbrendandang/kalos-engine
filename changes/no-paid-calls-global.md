### Added - global paid model switch

- `KALOS_NO_PAID_CALLS=1` (also `true`, `yes`, `on`) now blocks Anthropic
  assistant calls and keeps Anthropic normalization offline without an API call.
  Anthropic status reports unavailable. Remote OpenAI calls use the same shared
  flag check. Local servers and Ollama are unchanged.
