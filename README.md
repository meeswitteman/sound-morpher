# Sound Morpher

A desktop application for morphing between two audio samples in configurable discrete steps. Each step is an interpolated state between sound A and sound B, using a selectable morphing algorithm. The morph sequence can be previewed, played back in BPM-synchronized fashion, and exported as individual WAV files.

![Dark UI with spectrogram grid](resources/icons/app.svg)

---

## Features

- **7 morphing algorithms** — from simple crossfade to WORLD vocoder and Griffin-Lim
- **Spectrogram thumbnails** per morph step, computed on the fly
- **BPM-synchronized playback** with tap tempo and loop toggle; each step starts on the beat and the previous one rings out underneath it instead of being cut off
- **DTW Align** — optional Dynamic Time Warping preprocessing to align A and B before morphing, time-stretched through a phase vocoder so the alignment does not move either sound's pitch
- **Original endpoints** — the first step is always the untouched sound A and the last always the untouched sound B, heard in full
- **Level Match** — keeps loudness on a straight line from A to B, so intermediate steps do not sound thinner than the endpoints, with a look-ahead limiter catching any remaining peaks
- **Stretch to Fit** — time-stretch the shorter source to match the longer one instead of padding it with silence
- **Sweep** — instead of morphing the whole sound at once, let the morph travel through it step by step: Start → End, End → Start, or Center → Out. The Edge setting sets how wide the moving morph zone is
- **Live recording** — record directly into a source slot (mic or line-in)
- **Trim & volume** controls per source slot
- **Project files** — save and reload full sessions as `.smorph`
- **WAV export** — all steps as individual lossless WAV files
- **Plugin architecture** — add custom morph algorithms by dropping a Python file in `plugins/`

---

## Morphing Algorithms

| Plugin | Best for |
|--------|----------|
| **Crossfade** | Everything — equal-power (default) or linear volume blend |
| **Spectral FFT** | Textures and atmospheric sounds — interpolates STFT magnitude and phase |
| **Pitch Shift** | Melodic samples — moves both sounds onto a common interpolated pitch, then crossfades |
| **Granular** | Atmospheric morphs — rebuilds the sound from grains scattered between A and B, with position and pitch jitter |
| **Vocoder (LPC)** | Broadband audio — interpolates LPC spectral envelopes frame by frame via Line Spectral Frequencies |
| **WORLD Vocoder** | Voices and monophonic melodic samples — interpolates F0, spectral envelope, and aperiodicity using the WORLD speech synthesis framework |
| **Griffin-Lim** | Experimental / sci-fi textures — interpolates magnitude spectra and reconstructs phase via Griffin-Lim iteration |

### Magnitude and Phase modes

The spectral plugins (Spectral FFT, Griffin-Lim, WORLD Vocoder) expose how the
two spectra are combined:

- **Magnitude — `log`** (default) blends geometrically, so a partial present in A
  fades out as B's fades in. **`linear`** blends arithmetically, which leaves both
  spectra audible side by side: a spectral crossfade rather than a morph.
- **Phase — `shortest-arc`** (default) rotates A's phase toward B the short way
  round the circle. **`dominant`** takes the phase of the higher-weighted source.
  **`linear`** is the naive average; it averages straight through the ±π wrap and
  cancels partials that should reinforce, and is kept only for comparison.

Because geometric blending is quieter than arithmetic blending, leave **Level
Match** on when using `log`.

### Level Match and peaks

Loudness is measured as ITU-R BS.1770 integrated loudness (LUFS): K-weighted,
so a step with more low end is not mistaken for a louder one, and gated, so
silent stretches inside a step do not make it read quiet. The target runs in a
straight line in LUFS from A to B, which makes every step the same perceived
step; if one end is silent it runs linearly in amplitude instead, so the fade
still reaches nothing. Each step's gain is capped at 12 dB either way.
Spectral FFT between sources whose spectra hardly overlap (a dark pad into a
bright hiss, say) can lose more than that mid-sequence, and those steps then
stay somewhat quiet rather than having their noise floor pulled up.

After each step is scaled onto the loudness curve, anything over the ceiling
goes to a look-ahead limiter, which only acts where and when it has to. There is
no shared trim across the sequence: the endpoints stay at their original level
(see below), so trimming only the steps in between would put a dip into the
loudness line right next to them. A shared trim would also be the wrong tool for
the typical case, where one overshooting transient from spectral reconstruction
would drag the entire sequence down with it.

### Original endpoints

The first step is always sound A and the last always sound B, exactly as they
sit in the source slots: same samples, same length, heard in full. The engine
enforces this after everything else has run, so no plugin, no zero-padding, no
Stretch to Fit or DTW Align, and no level matching or limiting can change them.
When A and B differ in length, the endpoints keep their own length and the steps
in between take the longer one.

### Unequal lengths and tails

Without Stretch to Fit, the shorter sound is padded with silence, and a tail
can also simply decay away faster in one sound than in the other. A morph of a
sound with nothing is not meaningful: the geometric blend of Spectral FFT and
Griffin-Lim multiplied the sounding source away, and LPC / Source-Filter went
silent. Wherever one source sits 40 to 60 dB or more below the other, those
plugins now fall back, frame by frame, to a crossfade, so the steps in between
keep the longer sound's tail. Where both sources sound, the morph is unchanged.

### Pitch tracking

Pitch Shift takes a **Tracking** setting. `median` (default) finds one
representative pitch per sound and shifts by a constant interval — robust, and
the right choice for single-note samples. `dynamic` follows each sound's pitch
contour over time, so a melody or a vibrato morphs into the other's; it is the
better answer whenever the pitch actually moves, but detection is unreliable on
inharmonic or noisy material and the contour then warbles. On the bundled bell
samples, for instance, tracking spreads over a 2.6× range with several octave
jumps, which `median` sidesteps entirely.

### Formants

Pitch Shift takes a **Formants** setting. `preserve` (default) keeps each
sound's own resonances while its pitch moves, so a voice stays the same size
and an instrument keeps its body instead of turning into a "chipmunk". It
estimates each source's spectral envelope once (a true-envelope cepstral
estimate), and after every shift multiplies the spectrum by the ratio of the
source envelope to the same envelope stretched by the shift. On synthetic
vowels that brings the harmonic levels from 13-23 dB off the source's formant
curve to 3-10 dB. It roughly doubles the plugin's run time. `shift` lets the
resonances move with the pitch, as before.

### Transients

Spectral FFT and Griffin-Lim take a **Transients** setting. One FFT size cannot
suit both halves of a sound: long frames resolve partials but smear every
attack into pre-echo, short frames keep attacks sharp but blur the partials.
`preserve` (default) splits each source once into a tonal and a transient
layer (harmonic/percussive separation; the transient layer is the exact
remainder, so the two always sum back to the source). The tonal layer morphs at
the chosen FFT size, the transient layer at about 6 ms. On isolated drum hits
that cuts the pre-echo before each hit by 12-26 dB, and attacks rise in under
2 ms instead of 5-11 ms. Steady material comes out practically unchanged.
Spectral FFT takes about 3 s longer on 8 steps of 5 s stereo, Griffin-Lim about
7 s. `smear` morphs everything at one size, as before.

### Resampling and phase locking

Every path that changes pitch or timing (Pitch Shift in both modes, Granular's
pitch jitter, DTW Align) shares the same two building blocks:

- **Band-limited reads.** Fractional sample positions are read through a
  Kaiser-windowed sinc, or through soxr where the rate is constant. Linear
  interpolation used to cost about 7 dB at 15 kHz and let aliasing through
  almost unattenuated when reading faster than 1.0; the sinc lowers its cutoff
  with the rate and keeps aliasing around 90 dB down.
- **Phase-locked vocoder.** The phase vocoder uses identity phase locking
  (Laroche & Dolson): only spectral peaks advance their phase independently,
  and the bins around each peak keep their phase relationship to it. That
  removes most of the hollow, "phasey" smear of a plain vocoder, measured as
  10 to 18 dB less energy between the harmonics when stretching or shifting up.

### Channels

Both vocoders take a **Channels** setting. `stereo` (default) analyses and
synthesises each channel so the stereo image survives; the WORLD vocoder shares
one pitch track across channels, since estimating F0 per channel lets them drift
apart into a chorus. `mono` downmixes first and is roughly twice as fast.

### WORLD pitch detector

The WORLD vocoder takes a **Pitch detector** setting. `harvest` (default) keeps
notes voiced through noise, breath and vibrato. `dio` is about 7× faster to
analyse and just as accurate on clean recordings, but on noisy material it
dropped 10 to 20 % of the frames inside a note, which WORLD then rebuilds from
noise as crackle. The analysis runs once per morph, not once per step, so
Harvest's extra cost stays small.

---

## Loading sources

Every loaded sound is conditioned before it reaches a plugin:

- **Hot float WAVs are scaled, not clipped.** A 32-bit float file can peak
  above full scale. The whole file is scaled down to a peak of 1.0, which keeps
  the waveform intact, and the status bar says by how many dB.
- **DC offset is removed** with a zero-phase high-pass at 5 Hz. It has no gain
  at DC, costs 0.5 dB at 20 Hz and 0.2 dB at 30 Hz, and leaves the phase of
  kicks and bass untouched.
- **Sample-rate conversion** uses soxr at its highest quality setting.

---

## Export

Steps are written as `morph_step_01.wav` … `morph_step_NN.wav` at the project's
sample rate and bit depth. Choose 16-bit, 24-bit or 32-bit float under
**Project → Export Bit Depth**; new projects start from the last choice. Use
24-bit or float when the steps go into a DAW or sampler for further work.

16-bit exports get TPDF dither, which turns the
quantiser's signal-dependent distortion into an ordinary noise floor — most
audible on the fades and tails a morph sequence is full of. Toggle it under
**Project → Dither 16-bit Exports**; turn it off when the steps feed further
processing, so dither is applied only once at the very end. 24-bit and float
exports are never dithered, as the quantisation already sits below anything
audible.

The export bit depth only applies to the exported files. Audio inside a
`.smorph` project is always stored as 32-bit float, so saving and reopening a
project never costs quality.

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
5. **Preview** — click any step tile to hear it, or click **Play All** (Ctrl+Space) to play the full sequence. Each step rings out under the next, and Play All lights up while the sequence runs; clicking it again restarts from the first step. **Stop** (Escape) silences everything, including tails still ringing after the sequence ends.
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
