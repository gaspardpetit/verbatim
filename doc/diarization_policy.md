# Diarization policy grammar

Use a single string to describe how channels should be routed to diarization strategies:

- Clauses: `<targets>=<strategy>[?key=val&key=val...]`
- Targets: comma-separated channels or ranges; `*` is the fallback.
  - Examples: `1,2`, `0-2`, `1-3,5`, `*`
- Strategies: `pyannote`, `energy`, `channel`, `separate`.
- Params:
  - `speakers` (int): override speaker count for the strategy.
  - `channel` strategy only: `speaker` with `{idx}` placeholder, `offset` to shift idx.
  - `separate` strategy: accepts the same optional `speakers` hint and writes pyannote-separated stems (mono WAV files) that are referenced in the resulting VTTM.

Examples:

- `*=pyannote` — downmix all channels and run pyannote.
- `1,2=energy;3=pyannote;*=channel?speaker=HOST` — channels 1–2 run energy together; channel 3 runs pyannote; others are per-channel with labels `HOST`, `HOST_2`, `HOST_3`, etc.
- `0-2=channel;*=energy` — channels 0–2 are treated as fixed speakers; the rest use energy diarization.
- `0=separate;1=pyannote` — channel 0 is downmixed (if needed), separated with pyannote neural separation, and each resulting stem is written to disk while channel 1 is diarized with the standard pipeline.

Notes:
- Channels are zero-based.
- If a channel is not matched and no wildcard is provided, it is ignored (warning emitted).
- Separation clauses produce new audio files in the working directory and the VTTM emitted by the policy will reference those stem files via `audio` entries.
