# Sound Morpher — Software Requirements Document

**Version:** 1.0-draft  
**Date:** 2026-05-14  
**Status:** Draft

---

## 1. Project Overview

Sound Morpher is a cross-platform desktop application that enables music producers and sound designers to morph between two audio samples in a configurable number of discrete steps. Each step represents an interpolated state between sound A and sound B, using a selectable morphing algorithm. The resulting morph sequence can be previewed, played back in BPM-synchronized fashion, and exported as individual WAV files.

---

## 2. Goals & Non-Goals

### Goals
- Provide an intuitive, visually polished tool for audio morphing
- Support multiple morphing algorithms via a plugin architecture
- Enable BPM-synchronized playback of morph sequences
- Allow audio input via file selection or live recording
- Export morph steps as lossless WAV files
- Save and reload full sessions as `.smorph` project files

### Non-Goals (v1.0)
- DAW plugin (VST/AU) integration
- MIDI control or automation
- Real-time continuous morphing slider (post-v1.0)
- Cloud sync or sharing features
- Mobile or web versions

---

## 3. Target Audience

Primary users are **music producers and sound designers** who need granular control over sound transitions — for example in sample packs, game audio, film sound design, or experimental music production. Users are assumed to be technically literate but the UI must remain approachable.

---

## 4. Platform & Technology Stack

| Concern | Choice |
|---|---|
| Target OS | Windows, macOS, Linux (cross-platform) |
| UI Framework | Python + PySide6 (Qt6) |
| Audio processing | `numpy`, `scipy`, `librosa`, `sounddevice`, `soundfile` |
| Audio recording | `sounddevice` (microphone / line-in) |
| Project serialization | JSON + embedded or referenced WAV assets |
| Packaging | PyInstaller or cx_Freeze for distribution |

---

## 5. Audio Specifications

### Input
| Property | Specification |
|---|---|
| File formats (v1.0) | WAV |
| Sample rates | Any; normalized to project rate on load |
| Bit depth | Any; normalized to project bit depth on load |
| Channels | Mono and stereo supported |

### Project / Processing
| Property | Specification |
|---|---|
| Working sample rate | 44100 Hz or 48000 Hz (user-selectable per project) |
| Working bit depth | 16-bit (default); 24-bit optional per project |

### Output / Export
| Property | Specification |
|---|---|
| Format | WAV |
| Sample rate | Matches project sample rate |
| Bit depth | Matches project bit depth |
| Naming convention | `morph_step_01.wav` … `morph_step_NN.wav` |

---

## 6. Audio Input Methods

### 6.1 File Import
- User selects a WAV file via the OS file picker
- The file is **copied into the project** (embedded), so the project is self-contained
- The app normalizes sample rate and bit depth to the project settings on import
- File metadata (original filename, path, sample rate, bit depth, duration) is stored in the project file

### 6.2 Live Recording
- A **Record** button opens a recording panel
- User selects the input device (microphone or line-in) from a dropdown listing all available system audio input devices
- Configurable recording duration or manual stop
- Recorded audio is immediately saved as a WAV inside the project and assigned to slot A or B
- Input level meter displayed during recording

---

## 7. Morphing Architecture

### 7.1 Plugin System
Morphing algorithms are implemented as **plugins** — Python classes that conform to a common interface (`MorphPlugin`). The UI presents all registered plugins in a dropdown. New plugins can be added by dropping a file into the `plugins/` folder without modifying core code.

#### `MorphPlugin` Interface (conceptual)

```python
class MorphPlugin:
    name: str           # Display name in UI
    description: str    # Short description shown in UI
    parameters: list    # Declarative parameter definitions (sliders, dropdowns)

    def morph(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        steps: int,
        sample_rate: int,
        **params,
    ) -> list[np.ndarray]:
        """Return a list of `steps` audio arrays interpolating from A to B."""
        ...
```

### 7.2 Built-in Plugins (v1.0)

| Plugin | Description |
|---|---|
| **Crossfade** | Linear amplitude blend: step _i_ = `(1 - t) * A + t * B` where `t = i / (steps-1)` |
| **Spectral Morph (FFT)** | Interpolate magnitude spectra frame-by-frame via FFT; reconstruct with phase vocoder |
| **Pitch Shift Morph** | Shift the pitch of A towards B's detected fundamental over the step sequence |
| **Granular Synthesis Morph** | Slice both samples into grains; at each step blend the grain populations of A and B |

### 7.3 Step Count
- Configurable by the user: **2 to 32 steps** (inclusive)
- Default: 8 steps
- Steps include the endpoints: step 1 = 100% A, step N = 100% B

---

## 8. Playback

### 8.1 Individual Step Playback
- Each step is represented as a tile in the UI
- Clicking a tile plays that step once
- A stop button halts playback

### 8.2 Full Sequence Playback
- A **Play All** button plays all steps from step 1 to step N in order
- Playback can be stopped at any time

### 8.3 BPM-Synchronized Playback
- User sets a **BPM** value (20–300 BPM, default 120)
- User sets **N beats per step** — after N beats the next morph step begins playing (configurable, e.g. 1, 2, 4, 8 beats)
- Steps loop: after the last step, playback wraps back to step 1 (optional toggle)
- A beat/bar indicator (visual pulse) is shown during BPM playback
- Tap-tempo button to set BPM by tapping

---

## 9. Visualization

### 9.1 Spectrogram View
- Each morph step is displayed as a **spectrogram** (time × frequency, amplitude as color intensity)
- Spectrograms are computed using Short-Time Fourier Transform (STFT)
- Color map: visually distinct, perceptually uniform (e.g. viridis or a custom branded palette)
- Spectrograms are rendered asynchronously so they do not block the UI

### 9.2 Step Grid
- Steps are laid out in a horizontal scrollable row
- Each tile shows:
  - Step number
  - Spectrogram thumbnail
  - Duration (seconds)
  - Active/playing indicator (highlighted border or glow)
- Step A (first) and Step B (last) tiles are visually distinguished

### 9.3 Input Audio Panel
- Waveform display for sound A and sound B
- Playback button per input sound
- Duration and sample rate shown as metadata

---

## 10. Project Files

### 10.1 Format
- File extension: `.smorph`
- Structure: a **ZIP archive** containing:
  - `project.json` — all settings, parameters, step count, BPM, plugin choice
  - `audio/source_a.wav` — embedded copy of input sound A
  - `audio/source_b.wav` — embedded copy of input sound B
  - `audio/step_01.wav` … `audio/step_NN.wav` — pre-computed morph steps (optional cache)

### 10.2 Operations
- **New project** — clears current session
- **Open project** — load `.smorph` file via file picker
- **Save / Save As** — write current state to `.smorph`
- **Recent files** — last 10 projects listed in File menu

---

## 11. Export

- **Export Steps** action opens a folder picker
- All morph steps are exported as individual WAV files:  
  `morph_step_01.wav` … `morph_step_NN.wav`
- Export uses project sample rate and bit depth
- A progress dialog is shown during export

---

## 12. User Interface

### 12.1 Layout

```
┌────────────────────────────────────────────────────────────────────────┐
│  Menu bar: File | Project | Plugins | Help                             │
├──────────────┬─────────────────────────────────────────────────────────┤
│  Sound A     │  MORPH SETTINGS                                         │
│  [waveform]  │  Algorithm: [Spectral FFT ▼]   Steps: [8 ──────]        │
│  [▶] [●rec]  │  BPM: [120]  Beats/step: [4]  [Tap] [Loop ☐]           │
│              │─────────────────────────────────────────────────────────│
│  Sound B     │  STEP GRID  (horizontal scroll)                         │
│  [waveform]  │  [1:A] [2] [3] [4] [5] [6] [7] [8:B]                  │
│  [▶] [●rec]  │  (spectrogram thumbnails)                               │
├──────────────┴─────────────────────────────────────────────────────────┤
│  [▶ Play All]  [■ Stop]  [⟳ Recompute]  [↓ Export Steps]              │
│  Beat indicator: ● ● ● ●                                               │
└────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Visual Design Principles
- Dark theme (professional audio-tool aesthetic)
- Accent color: one prominent brand color (e.g. teal or amber) for active/interactive elements
- Responsive layout: resizable window, step grid scrollable
- No clutter: advanced plugin parameters shown in a collapsible panel

### 12.3 Recording Panel (modal or side panel)
- Input device selector
- Level meter (VU-style)
- Record / Stop buttons
- Preview recorded audio before accepting
- Assign to: Sound A / Sound B

---

## 13. Plugin Parameter UI

Each plugin can declare parameters that are automatically rendered in the UI:

| Parameter type | Widget |
|---|---|
| Float range | Slider with numeric readout |
| Integer range | Spin box |
| Enum / choice | Dropdown |
| Boolean | Toggle switch |

Parameters are stored per-project alongside the plugin name.

---

## 14. Error Handling & Edge Cases

| Scenario | Behavior |
|---|---|
| Sound A and B have different durations | Shorter sample is zero-padded or looped to match longer one (user-configurable) |
| Sound A and B have different sample rates | Both resampled to project sample rate before processing |
| Plugin throws an exception | Error shown inline in step tile; other steps unaffected |
| Recording device unavailable | Friendly error dialog with device troubleshooting hint |
| Unsupported WAV variant (e.g. 32-bit float) | Converted to project bit depth on load with user notification |

---

## 15. Non-Functional Requirements

| Property | Requirement |
|---|---|
| Performance | Morph computation for 8 steps of ≤5s audio completes in <10 seconds |
| UI responsiveness | Audio computation runs on background threads; UI never blocks |
| Memory | Application idle memory usage <200 MB |
| Startup time | Cold start <5 seconds |
| Accessibility | Keyboard navigation for all primary actions |

---

## 16. Versioning & Extensibility Roadmap

| Version | Scope |
|---|---|
| **v1.0** | Core morphing (4 plugins), WAV I/O, BPM playback, spectrogram view, project files, export |
| **v1.1** | Additional input formats (MP3, FLAC, OGG); drag-and-drop file import |
| **v1.2** | Real-time slider preview (continuous morph position); waveform overlay on spectrogram |
| **v2.0** | VST/AU plugin wrapper; MIDI clock sync; additional community plugins |

---

## 17. Decisions (formerly Open Questions)

| Question | Decision |
|----------|----------|
| Cache pre-computed step WAVs inside `.smorph`? | **Yes** — step WAVs are stored in the archive; recomputation only needed when settings change |
| Support multi-channel audio (> stereo)? | **No** — mono and stereo only, no surround/multichannel |
| Headless / CLI mode? | **No** — desktop UI only |

---

*Generated from requirements session on 2026-05-14. Open questions resolved 2026-05-14.*
