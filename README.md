# Sound Morpher

A desktop application for morphing between two audio samples in configurable discrete steps. Each step is an interpolated state between sound A and sound B, using a selectable morphing algorithm. The morph sequence can be previewed, played back in BPM-synchronized fashion, and exported as individual WAV files.

![Dark UI with spectrogram grid](resources/icons/app.svg)

---

## Features

- **6 morphing algorithms** — from simple crossfade to WORLD vocoder
- **Spectrogram thumbnails** per morph step, computed on the fly
- **BPM-synchronized playback** with tap tempo and loop toggle
- **Live recording** — record directly into a source slot (mic or line-in)
- **Trim & volume** controls per source slot
- **Project files** — save and reload full sessions as `.smorph`
- **WAV export** — all steps as individual lossless WAV files
- **Plugin architecture** — add custom morph algorithms by dropping a Python file in `plugins/`

---

## Morphing Algorithms

| Plugin | Best for |
|--------|----------|
| **Crossfade** | Everything — simple linear or equal-power volume blend |
| **Spectral FFT** | Textures and atmospheric sounds — interpolates STFT magnitude and phase |
| **Pitch Shift** | Melodic samples — shifts pitch while preserving timbre |
| **Granular** | Atmospheric morphs — blends overlapping grains from both sources |
| **Vocoder (LPC)** | Broadband audio — interpolates LPC spectral envelopes frame by frame |
| **WORLD Vocoder** | Voices and monophonic melodic samples — interpolates F0, spectral envelope, and aperiodicity using the WORLD speech synthesis framework |

---

## Requirements

- Python 3.11 or newer
- Windows, macOS, or Linux

### Dependencies

```
PySide6 >= 6.6
numpy >= 1.26
scipy >= 1.12
librosa >= 0.10
sounddevice >= 0.4
soundfile >= 0.12
pyworld >= 0.3.5   # for WORLD Vocoder
```

---

## Installation

```bash
git clone https://github.com/meeswitteman/sound-morpher.git
cd sound-morpher
pip install -r requirements.txt
pip install pyworld          # optional — only needed for WORLD Vocoder
```

---

## Running

```bash
python main.py
```

---

## Usage

1. **Load sounds** — drag a WAV file onto slot A and slot B, or click the slot to browse. You can also record directly via the microphone button.
2. **Set steps** — choose how many morph steps (2–32) using the steps spinner.
3. **Choose algorithm** — select a morphing algorithm from the dropdown and adjust its parameters.
4. **Compute** — click **Recompute** to generate all steps. Spectrogram thumbnails appear immediately.
5. **Preview** — click any step tile to hear it, or use the transport buttons to play the full sequence.
6. **BPM sync** — set the BPM and beats-per-step to lock playback to your project tempo. Use **Tap** to measure tempo from a beat.
7. **Export** — click **Export WAVs** to save all steps as `morph_step_01.wav` … `morph_step_NN.wav`.
8. **Save session** — use **File → Save** to write a `.smorph` project file that embeds both source WAVs and all settings.

---

## Project File Format

`.smorph` files are ZIP archives containing:

```
project.json   — metadata, algorithm settings, BPM, step count
audio_a.wav    — embedded source A
audio_b.wav    — embedded source B
```

---

## Writing a Custom Plugin

Create a file in `plugins/` that subclasses `MorphPlugin`:

```python
from plugins.base import MorphPlugin, PluginParam, match_lengths
import numpy as np

class MyPlugin(MorphPlugin):
    name = "My Plugin"
    description = "Does something interesting."
    parameters = [
        PluginParam(name="amount", label="Amount", type="float",
                    default=0.5, min_val=0.0, max_val=1.0),
    ]

    def morph(self, audio_a, audio_b, steps, sample_rate, amount=0.5, **_):
        a, b = match_lengths(audio_a, audio_b)
        result = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            result.append(((1 - t) * a + t * b).astype(np.float32))
        return result
```

The plugin is discovered and registered automatically at startup — no further changes needed.

---

## Running Tests

```bash
pip install pytest
pytest
```

---

## License

MIT
