### Fixed - remote OpenAI provider availability

- Remote OpenAI-compatible servers report unavailable until `KALOS_OPENAI_ALLOW_REMOTE`
  is enabled. Availability and call-time refusal use the same policy check.
  `disabled_by` names the blocking policy variable; `missing_env` remains credentials only.
